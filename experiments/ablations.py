import glob
import heapq
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

from analysis import DATA, DATASETS, OUT, SEED, bootstrap_ci, classify, conj_soundness, load_attempts, load_dir, soundness, task_inv, task_score
from invariants import FAMILIES, as_grid

SAFE = ["palette_exact", "palette_subset", "hist_equal", "nonbg_ge", "keep_nonbg", "input_in_output"]
ADD_ONE = ["shape", "keep_bg", "nonbg_equal", "nonbg_le", "palette_keep", "sym_lr"]
N_DRAWS = int(os.environ.get("N_DRAWS", "200"))
SIZES = (2, 4, 8, 16, 32)
INF = float("inf")


def family_sets():
    sets = {"none": [], "safe6": SAFE, "all16": FAMILIES}
    for f in SAFE:
        sets[f"safe6-{f}"] = [g for g in SAFE if g != f]
    for f in ADD_ONE:
        sets[f"safe6+{f}"] = SAFE + [f]
    sets["shape_only"] = ["shape"]
    return sets


def flagged(res, fams):
    return not all(res.get(f, True) for f in fams)


def load_conceptarc():
    tasks = {}
    for f in sorted(glob.glob(os.path.join(DATA, "ConceptARC", "corpus", "*", "*.json"))):
        concept = os.path.basename(os.path.dirname(f))
        with open(f) as fh:
            tasks[f"{concept}/{os.path.basename(f)[:-5]}"] = json.load(fh)
    return tasks


def by_k_table(rows, fams):
    out = defaultdict(lambda: {"n": 0, "gt_fail": 0, "decoys": 0, "decoy_pass": 0})
    for r in rows:
        k = str(min(r["k"], 5))
        o = out[k]
        o["n"] += 1
        o["gt_fail"] += int(flagged(r["res"], fams))
        for d in r["decoy"]:
            o["decoys"] += 1
            o["decoy_pass"] += int(not flagged(d, fams))
    return dict(sorted(out.items()))


def soundness_section(sets):
    res = {}
    datasets = {k: load_dir(v) for k, v in DATASETS.items()}
    datasets["conceptarc"] = load_conceptarc()
    for name, tasks in datasets.items():
        rows, fam = soundness(name, tasks)
        entry = {"tasks": len(tasks), "test_outputs": len(rows), "sets": {}, "by_k": {}}
        for sname, fams in sets.items():
            if not fams:
                continue
            cs = conj_soundness(rows, fams)
            entry["sets"][sname] = {"gt_pass_rate": cs["gt_pass_rate"], "decoy_pass_rate": cs["decoy_pass_rate"], "failures": cs["failures"]}
        for sname in ("safe6", "all16"):
            entry["by_k"][sname] = by_k_table(rows, sets[sname])
        if name == "conceptarc":
            per_concept = defaultdict(lambda: [0, 0])
            for r in rows:
                c = r["task"].split("/")[0]
                per_concept[c][0] += 1
                per_concept[c][1] += int(not flagged(r["res"], SAFE))
            entry["per_concept"] = {c: {"n": v[0], "pass": v[1]} for c, v in sorted(per_concept.items())}
            entry["families"] = {f: {"active": fam[f]["active"], "soundness": fam[f]["soundness"], "decoy_pass": fam[f]["decoy_pass"]} for f in FAMILIES}
        res[name] = entry
        print(f"== soundness {name}: {len(tasks)} tasks, {len(rows)} outputs", flush=True)
        for sname, v in entry["sets"].items():
            print(f"   {sname:22s} gt_pass={v['gt_pass_rate']:.4f} decoy_pass={v['decoy_pass_rate']:.3f} failures={len(v['failures'])}")
        for sname, t in entry["by_k"].items():
            print(f"   by_k {sname}: " + " ".join(f"k{k}: n={v['n']} fail={v['gt_fail']} decoy_pass={v['decoy_pass']/max(1,v['decoys']):.3f}" for k, v in t.items()))
        if "per_concept" in entry:
            print("   per concept: " + " ".join(f"{c}={v['pass']}/{v['n']}" for c, v in entry["per_concept"].items()))
    return res


def attempt_section(eval_tasks, cands, sets):
    valid = [c for c in cands if c["grid"] is not None]
    wrong = [c for c in valid if c["cls"] != "correct"]
    right = [c for c in valid if c["cls"] == "correct"]
    ks = {tid: len(t["train"]) for tid, t in eval_tasks.items()}
    out = {"valid": len(valid), "wrong": len(wrong), "correct": len(right), "families": {}, "sets": {}, "by_k": {}, "by_class": {}}
    for f in FAMILIES:
        out["families"][f] = {"recall": sum(1 for c in wrong if flagged(c["res"], [f])) / len(wrong),
                              "false_flag": sum(1 for c in right if flagged(c["res"], [f])) / len(right),
                              "active_share": sum(1 for c in valid if f in c["res"]) / len(valid)}
    classes = ["near_miss", "wrong_content", "wrong_palette", "wrong_shape"]
    for sname, fams in sets.items():
        if not fams:
            continue
        out["sets"][sname] = {"recall": sum(1 for c in wrong if flagged(c["res"], fams)) / len(wrong),
                              "false_flag": sum(1 for c in right if flagged(c["res"], fams)) / len(right)}
        out["by_class"][sname] = {cl: sum(1 for c in wrong if c["cls"] == cl and flagged(c["res"], fams)) / max(1, sum(1 for c in wrong if c["cls"] == cl)) for cl in classes}
    for sname in ("safe6", "all16"):
        t = defaultdict(lambda: {"wrong": 0, "wrong_flagged": 0, "correct": 0, "correct_flagged": 0})
        for c in valid:
            k = str(min(ks[c["task"]], 5))
            fl = flagged(c["res"], sets[sname])
            if c["cls"] == "correct":
                t[k]["correct"] += 1
                t[k]["correct_flagged"] += int(fl)
            else:
                t[k]["wrong"] += 1
                t[k]["wrong_flagged"] += int(fl)
        out["by_k"][sname] = dict(sorted(t.items()))
    print(f"== attempts: valid={len(valid)} wrong={len(wrong)} correct={len(right)}", flush=True)
    for f, v in out["families"].items():
        print(f"   family {f:16s} recall={v['recall']:.3f} false_flag={v['false_flag']:.4f} active={v['active_share']:.3f}")
    for s, v in out["sets"].items():
        print(f"   set {s:22s} recall={v['recall']:.3f} false_flag={v['false_flag']:.4f} by_class=" + " ".join(f"{cl}={x:.3f}" for cl, x in out["by_class"][s].items()))
    for s, t in out["by_k"].items():
        print(f"   by_k {s}: " + " ".join(f"k{k}: recall={v['wrong_flagged']/max(1,v['wrong']):.3f} ff={v['correct_flagged']/max(1,v['correct']):.4f} (n={v['wrong']}/{v['correct']})" for k, v in t.items()))
    return out


def build_groups(eval_tasks, cands):
    groups = defaultdict(dict)
    for c in cands:
        if c["grid"] is None:
            continue
        h = c["grid"].tobytes() + bytes(c["grid"].shape)
        g = groups[(c["task"], c["test"])].setdefault(h, {"models": set(), "first": set(), "correct": c["cls"] == "correct", "res": c["res"]})
        g["models"].add(c["model"])
        if c["attempt"] == "attempt_1":
            g["first"].add(c["model"])
    keys = [(tid, ti) for tid, t in eval_tasks.items() for ti in range(len(t["test"]))]
    return keys, {k: list(v.values()) for k, v in groups.items()}


def configs(sets):
    cfg = [("vote", None, 0.0)]
    for sname, fams in sets.items():
        if not fams:
            continue
        cfg.append((f"{sname}|lam1", fams, 1.0))
        cfg.append((f"{sname}|hard", fams, INF))
    for lam in (0.5, 2.0):
        cfg.append((f"safe6|lam{lam}", SAFE, lam))
    return cfg


def evaluate(keys, groups, model_set, cfg, rng):
    ok = {name: {} for name, _, _ in cfg}
    ok["oracle"] = {}
    for k in keys:
        gl = []
        for g in groups.get(k, []):
            n = len(g["models"] & model_set)
            if n:
                gl.append((n, len(g["first"] & model_set), g["correct"], g["res"], rng.random()))
        ok["oracle"][k] = any(g[2] for g in gl)
        for name, fams, lam in cfg:
            if fams is None:
                top = heapq.nsmallest(2, gl, key=lambda g: (-g[0], -g[1], g[4]))
            elif lam == INF:
                top = heapq.nsmallest(2, gl, key=lambda g: (flagged(g[3], fams), -g[0], -g[1], g[4]))
            else:
                top = heapq.nsmallest(2, gl, key=lambda g: (-(g[0] - lam * flagged(g[3], fams)), -g[0], -g[1], g[4]))
            ok[name][k] = any(g[2] for g in top)
    return ok


def summarize(draw_results, keys):
    names = list(draw_results[0].keys())
    n_out = Counter(k[0] for k in keys)
    out = {}
    for name in names:
        scores = [task_score(d[name]) for d in draw_results]
        out[name] = {"pass2": float(np.mean(scores))}
        if name != "vote":
            diffs = [task_score(d[name]) - task_score(d["vote"]) for d in draw_results]
            gained = [sum(1 for k in keys if d[name][k] and not d["vote"][k]) for d in draw_results]
            lost = [sum(1 for k in keys if d["vote"][k] and not d[name][k]) for d in draw_results]
            per_task = defaultdict(float)
            for d in draw_results:
                for k in keys:
                    per_task[k[0]] += (float(d[name][k]) - float(d["vote"][k])) / (len(draw_results) * n_out[k[0]])
            out[name].update({"gain": float(np.mean(diffs)), "wins": int(sum(1 for x in diffs if x > 1e-9)), "losses": int(sum(1 for x in diffs if x < -1e-9)),
                              "outputs_gained": float(np.mean(gained)), "outputs_lost": float(np.mean(lost)),
                              "ci": bootstrap_ci(list(per_task.values()))})
    return out


def pooled_section(eval_tasks, cands, models, sets):
    keys, groups = build_groups(eval_tasks, cands)
    cfg = configs(sets)
    rng = random.Random(SEED)
    res = {}
    for n in SIZES:
        draws = [evaluate(keys, groups, set(rng.sample(models, n)), cfg, rng) for _ in range(N_DRAWS)]
        res[f"random_{n}"] = summarize(draws, keys)
        print(f"== pooled n={n}: vote={res[f'random_{n}']['vote']['pass2']:.4f} oracle={res[f'random_{n}']['oracle']['pass2']:.4f}", flush=True)
        for name, v in res[f"random_{n}"].items():
            if "gain" in v:
                print(f"   {name:28s} gain={v['gain']*100:+.2f} ci=[{v['ci'][1]*100:+.2f},{v['ci'][2]*100:+.2f}] wins={v['wins']} losses={v['losses']} outputs +{v['outputs_gained']:.2f}/-{v['outputs_lost']:.2f}")
    draws = [evaluate(keys, groups, set(models), cfg, rng) for _ in range(20)]
    res["all_models"] = summarize(draws, keys)
    print(f"== pooled all: vote={res['all_models']['vote']['pass2']:.4f} oracle={res['all_models']['oracle']['pass2']:.4f}", flush=True)
    for name, v in res["all_models"].items():
        if "gain" in v:
            print(f"   {name:28s} gain={v['gain']*100:+.2f} wins={v['wins']} losses={v['losses']} outputs +{v['outputs_gained']:.2f}/-{v['outputs_lost']:.2f}")
    pass2 = {}
    for m in models:
        ok = {}
        for c in cands:
            if c["model"] == m:
                ok[(c["task"], c["test"])] = ok.get((c["task"], c["test"]), False) or c["cls"] == "correct"
        pass2[m] = task_score(ok) if ok else 0.0
    ranked = sorted(models, key=lambda m: -pass2[m])
    top_cfg = [("vote", None, 0.0), ("safe6|lam1", SAFE, 1.0), ("safe6|hard", SAFE, INF), ("all16|lam1", FAMILIES, 1.0), ("all16|hard", FAMILIES, INF)]
    for n in (2, 4, 8, 16):
        draws = [evaluate(keys, groups, set(ranked[:n]), top_cfg, rng) for _ in range(20)]
        res[f"top_{n}"] = summarize(draws, keys)
        res[f"top_{n}"]["models"] = ranked[:n]
        print(f"== top-{n} systems: vote={res[f'top_{n}']['vote']['pass2']:.4f} oracle={res[f'top_{n}']['oracle']['pass2']:.4f} " + " ".join(f"{k}={v['gain']*100:+.2f}" for k, v in res[f"top_{n}"].items() if "gain" in v), flush=True)
    return res


def theory_section(pooled, eps, n_out):
    out = {}
    for key, v in pooled.items():
        if "safe6|hard" not in v:
            continue
        vote = v["vote"]["pass2"]
        out[key] = {"vote_pass2": vote, "eps": eps, "predicted_loss_bound": eps * vote,
                    "hard_loss": v["safe6|hard"]["outputs_lost"] / n_out, "hard_gain": v["safe6|hard"]["outputs_gained"] / n_out,
                    "lam1_loss": v["safe6|lam1"]["outputs_lost"] / n_out, "lam1_gain": v["safe6|lam1"]["outputs_gained"] / n_out,
                    "gap_closed_lam1": (v["safe6|lam1"]["pass2"] - vote) / max(1e-9, v["oracle"]["pass2"] - vote)}
        o = out[key]
        print(f"   theory {key:12s} vote={vote:.3f} bound={o['predicted_loss_bound']*100:.2f} hard loss={o['hard_loss']*100:.2f} gain={o['hard_gain']*100:.2f} | lam1 loss={o['lam1_loss']*100:.2f} gain={o['lam1_gain']*100:.2f} gap_closed={o['gap_closed_lam1']*100:.1f}%")
    return out


def main():
    attempt_set = os.environ.get("ATTEMPT_SET", "v2")
    eval_key, hf_dir = {"v2": ("agi2_eval", "hf_v2_eval"), "v1": ("agi1_eval", "hf_v1_eval")}[attempt_set]
    sets = family_sets()
    summary = {"seed": SEED, "n_draws": N_DRAWS, "attempt_set": attempt_set, "sets": sets}
    summary["soundness"] = soundness_section(sets)
    ev = load_dir(DATASETS[eval_key])
    models, cands = load_attempts(ev, hf_dir)
    gts = {(tid, ti): as_grid(p["output"]) for tid, t in ev.items() for ti, p in enumerate(t["test"])}
    inv = {}
    for c in cands:
        c["cls"] = classify(c["grid"], gts[(c["task"], c["test"])])
        if c["grid"] is None:
            c["res"] = None
            continue
        if c["task"] not in inv:
            inv[c["task"]] = task_inv(ev[c["task"]])
        c["res"] = inv[c["task"]].check(ev[c["task"]]["test"][c["test"]]["input"], c["grid"])
    summary["attempts"] = attempt_section(ev, cands, sets)
    summary["pooled"] = pooled_section(ev, cands, models, sets)
    n_out = sum(len(t["test"]) for t in ev.values())
    eps = 1 - summary["soundness"][eval_key]["sets"]["safe6"]["gt_pass_rate"]
    summary["theory"] = theory_section(summary["pooled"], eps, n_out)
    summary["n_outputs"] = n_out
    name = "ablations.json" if attempt_set == "v2" else f"ablations_{attempt_set}.json"
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
