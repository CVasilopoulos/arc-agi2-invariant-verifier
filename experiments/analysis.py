import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

from invariants import FAMILIES, TaskInvariants, as_grid

ROOT = os.environ.get("ARC_ROOT", "/work")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results")
SEED = 20260916
MIN_TASKS_PER_MODEL = 90
SAFE_THRESHOLD = 0.99
SAFE_MIN_ACTIVE = 30

DATASETS = {
    "agi2_train": os.path.join(DATA, "ARC-AGI-2/data/training"),
    "agi2_eval": os.path.join(DATA, "ARC-AGI-2/data/evaluation"),
    "agi1_eval": os.path.join(DATA, "ARC-AGI-1/data/evaluation"),
}


def load_dir(path):
    tasks = {}
    for f in sorted(glob.glob(os.path.join(path, "*.json"))):
        with open(f) as fh:
            tasks[os.path.basename(f)[:-5]] = json.load(fh)
    return tasks


def task_inv(task):
    return TaskInvariants([(p["input"], p["output"]) for p in task["train"]])


def gt_rows(item):
    tid, task, decoys = item
    inv = task_inv(task)
    rows = []
    for ti, p in enumerate(task["test"]):
        res = inv.check(p["input"], p["output"])
        decoy_res = [inv.check(p["input"], d) for d in decoys]
        rows.append({"task": tid, "test": ti, "k": inv.k, "res": res, "decoy": decoy_res})
    return rows


def soundness(dataset_name, tasks, n_decoys=20):
    rng = random.Random(SEED)
    outputs = [(tid, p["output"]) for tid, t in tasks.items() for p in t["test"]]
    items = []
    for tid, t in tasks.items():
        pool = [o for otid, o in outputs if otid != tid]
        items.append((tid, t, rng.sample(pool, min(n_decoys, len(pool)))))
    with Pool(8) as pool:
        rows = [r for rs in pool.map(gt_rows, items, chunksize=8) for r in rs]
    fam = {}
    for f in FAMILIES:
        act = [r for r in rows if f in r["res"]]
        ok = sum(1 for r in act if r["res"][f])
        dec = [d[f] for r in act for d in r["decoy"] if f in d]
        by_k = defaultdict(lambda: [0, 0])
        for r in act:
            kk = min(r["k"], 5)
            by_k[kk][0] += 1
            by_k[kk][1] += int(r["res"][f])
        fam[f] = {"active": len(act), "sound": ok, "soundness": ok / len(act) if act else None,
                  "decoy_pass": (sum(dec) / len(dec)) if dec else None, "by_k": {str(k): v for k, v in sorted(by_k.items())}}
    return rows, fam


def conj_soundness(rows, families):
    n = len(rows)
    ok = sum(1 for r in rows if all(r["res"].get(f, True) for f in families))
    active = sum(1 for r in rows if any(f in r["res"] for f in families))
    fails = [(r["task"], r["test"], [f for f in families if not r["res"].get(f, True)]) for r in rows if not all(r["res"].get(f, True) for f in families)]
    dec_total = sum(len(r["decoy"]) for r in rows)
    dec_pass = sum(1 for r in rows for d in r["decoy"] if all(d.get(f, True) for f in families))
    return {"test_outputs": n, "any_active": active, "gt_pass_all": ok, "gt_pass_rate": ok / n,
            "decoy_pass_rate": dec_pass / dec_total, "failures": fails}


def valid_grid(a):
    if not isinstance(a, list) or not a or not all(isinstance(r, list) and r for r in a):
        return None
    w = len(a[0])
    if len(a) > 30 or w > 30 or any(len(r) != w for r in a):
        return None
    for r in a:
        for v in r:
            if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v > 9:
                return None
    return np.array(a, dtype=np.int16)


def load_attempts(eval_tasks, hf_dir):
    root = os.path.join(DATA, hf_dir)
    models = []
    for m in sorted(os.listdir(root)):
        p = os.path.join(root, m)
        if not os.path.isdir(p) or m.startswith("."):
            continue
        files = [f for f in os.listdir(p) if f.endswith(".json") and f[:-5] in eval_tasks]
        if len(files) >= MIN_TASKS_PER_MODEL:
            models.append(m)
    cands = []
    for m in models:
        for tid, task in eval_tasks.items():
            f = os.path.join(root, m, tid + ".json")
            if not os.path.exists(f):
                continue
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except Exception:
                d = []
            for ti in range(len(task["test"])):
                entry = d[ti] if isinstance(d, list) and ti < len(d) and isinstance(d[ti], dict) else {}
                for att in ("attempt_1", "attempt_2"):
                    a = entry.get(att) or {}
                    ans = a.get("answer") if isinstance(a, dict) else None
                    cands.append({"model": m, "task": tid, "test": ti, "attempt": att, "grid": valid_grid(ans)})
    return models, cands


def classify(g, gt):
    if g is None:
        return "invalid"
    if g.shape != gt.shape:
        return "wrong_shape"
    if np.array_equal(g, gt):
        return "correct"
    if set(np.unique(g).tolist()) != set(np.unique(gt).tolist()):
        return "wrong_palette"
    acc = float((g == gt).mean())
    return "near_miss" if acc >= 0.95 else "wrong_content"


def attempt_analysis(eval_tasks, safe, hf_dir):
    models, cands = load_attempts(eval_tasks, hf_dir)
    gts = {(tid, ti): as_grid(p["output"]) for tid, t in eval_tasks.items() for ti, p in enumerate(t["test"])}
    inv_cache = {}
    for c in cands:
        c["cls"] = classify(c["grid"], gts[(c["task"], c["test"])])
    for c in cands:
        if c["grid"] is None:
            c["res"] = None
            continue
        key = c["task"]
        if key not in inv_cache:
            inv_cache[key] = task_inv(eval_tasks[key])
        c["res"] = inv_cache[key].check(eval_tasks[key]["test"][c["test"]]["input"], c["grid"])
        c["flag"] = not all(c["res"].get(f, True) for f in safe)
    tax = Counter(c["cls"] for c in cands)
    flag_by_cls = defaultdict(lambda: [0, 0])
    for c in cands:
        if c["grid"] is None:
            continue
        flag_by_cls[c["cls"]][0] += 1
        flag_by_cls[c["cls"]][1] += int(c["flag"])
    fam_veto = {}
    for f in FAMILIES:
        wrong = [c for c in cands if c["grid"] is not None and c["cls"] != "correct" and f in c["res"]]
        right = [c for c in cands if c["grid"] is not None and c["cls"] == "correct" and f in c["res"]]
        fam_veto[f] = {"wrong_active": len(wrong), "wrong_vetoed": sum(1 for c in wrong if not c["res"][f]),
                       "correct_active": len(right), "correct_vetoed": sum(1 for c in right if not c["res"][f])}
    return models, cands, tax, flag_by_cls, fam_veto


def per_task(correct_by_test):
    per = defaultdict(list)
    for (tid, ti), ok in correct_by_test.items():
        per[tid].append(ok)
    return {tid: float(np.mean(v)) for tid, v in per.items()}


def bootstrap_ci(values, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def task_score(correct_by_test):
    per_task = defaultdict(list)
    for (tid, ti), ok in correct_by_test.items():
        per_task[tid].append(ok)
    return sum(np.mean(v) for v in per_task.values()) / len(per_task)


def per_model_order(eval_tasks, models, cands):
    idx = defaultdict(dict)
    for c in cands:
        idx[(c["model"], c["task"], c["test"])][c["attempt"]] = c
    out = {}
    all_keys = [(tid, ti) for tid, t in eval_tasks.items() for ti in range(len(t["test"]))]
    for m in models:
        p1, p1v, p2 = {}, {}, {}
        swapped = fixed = broken = 0
        covered = set(tid for (mm, tid, ti) in idx if mm == m)
        for tid, ti in all_keys:
            if tid not in covered:
                continue
            e = idx.get((m, tid, ti), {})
            a1, a2 = e.get("attempt_1"), e.get("attempt_2")
            c1 = bool(a1 and a1["cls"] == "correct")
            c2 = bool(a2 and a2["cls"] == "correct")
            p1[(tid, ti)] = c1
            p2[(tid, ti)] = c1 or c2
            f1 = (a1 is None) or a1["grid"] is None or a1["flag"]
            f2 = (a2 is None) or a2["grid"] is None or a2["flag"]
            p1v[(tid, ti)] = c2 if (f1 and not f2) else c1
            if f1 and not f2 and c2 and not c1:
                fixed += 1
            if f1 and not f2 and c1 and not c2:
                broken += 1
            if f1 and not f2:
                swapped += 1
        out[m] = {"tasks": len(covered), "pass1": task_score(p1), "pass1_verifier": task_score(p1v), "pass2": task_score(p2),
                  "swapped": swapped, "fixed": fixed, "broken": broken,
                  "delta_by_task": {tid: v - per_task(p1)[tid] for tid, v in per_task(p1v).items()}}
    return out


def pooled_selection(eval_tasks, cands, model_set, safe, rng):
    by_key = defaultdict(list)
    ms = set(model_set)
    for c in cands:
        if c["model"] in ms and c["grid"] is not None:
            by_key[(c["task"], c["test"])].append(c)
    ok = {name: {} for name in ("vote", "hard", "tiebreak", "soft1", "hedge", "oracle")}
    for tid, t in eval_tasks.items():
        for ti in range(len(t["test"])):
            pool = by_key.get((tid, ti), [])
            groups = {}
            for c in pool:
                h = c["grid"].tobytes() + bytes(c["grid"].shape)
                g = groups.setdefault(h, {"models": set(), "first": 0, "correct": c["cls"] == "correct", "flag": c["flag"], "r": rng.random()})
                g["models"].add(c["model"])
                if c["attempt"] == "attempt_1":
                    g["first"] += 1
            gl = list(groups.values())
            orders = {
                "vote": sorted(gl, key=lambda g: (-len(g["models"]), -g["first"], g["r"])),
                "hard": sorted(gl, key=lambda g: (g["flag"], -len(g["models"]), -g["first"], g["r"])),
                "tiebreak": sorted(gl, key=lambda g: (-len(g["models"]), g["flag"], -g["first"], g["r"])),
                "soft1": sorted(gl, key=lambda g: (-(len(g["models"]) - int(g["flag"])), -len(g["models"]), -g["first"], g["r"])),
            }
            v = orders["vote"]
            hedge = v[:1]
            rest = [g for g in v[1:] if not g["flag"]] or v[1:]
            hedge = hedge + rest[:1]
            orders["hedge"] = hedge
            for name, order in orders.items():
                ok[name][(tid, ti)] = any(g["correct"] for g in order[:2])
            ok["oracle"][(tid, ti)] = any(g["correct"] for g in gl)
    return {name: task_score(v) for name, v in ok.items()}, {name: per_task(v) for name, v in ok.items()}


def main():
    attempt_set = os.environ.get("ATTEMPT_SET", "v2")
    eval_key, hf_dir = {"v2": ("agi2_eval", "hf_v2_eval"), "v1": ("agi1_eval", "hf_v1_eval")}[attempt_set]
    os.makedirs(OUT, exist_ok=True)
    tasks = {k: load_dir(v) for k, v in DATASETS.items()}
    summary = {"seed": SEED, "attempt_set": attempt_set}
    rows_all = {}
    for name, ts in tasks.items():
        rows, fam = soundness(name, ts)
        rows_all[name] = rows
        summary[f"soundness_{name}"] = fam
        print(f"== {name}: {len(ts)} tasks, {len(rows)} test outputs", flush=True)
        for f in FAMILIES:
            v = fam[f]
            sd = "-" if v["soundness"] is None else f"{v['soundness']:.3f}"
            d = "-" if v["decoy_pass"] is None else f"{v['decoy_pass']:.3f}"
            print(f"   {f:16s} active={v['active']:5d} sound={sd} decoy_pass={d} by_k={v['by_k']}")
    tr = summary["soundness_agi2_train"]
    safe = [f for f in FAMILIES if tr[f]["active"] >= SAFE_MIN_ACTIVE and tr[f]["soundness"] is not None and tr[f]["soundness"] >= SAFE_THRESHOLD]
    summary["safe_families"] = safe
    print("SAFE (chosen on ARC-AGI-2 training):", safe, flush=True)
    for name, rows in rows_all.items():
        cs = conj_soundness(rows, safe)
        summary[f"conj_{name}"] = cs
        by_k = defaultdict(lambda: [0, 0])
        for r in rows:
            by_k[min(r["k"], 5)][0] += 1
            by_k[min(r["k"], 5)][1] += int(all(r["res"].get(f, True) for f in safe))
        summary[f"conj_by_k_{name}"] = {str(k): v for k, v in sorted(by_k.items())}
        print(f"   conj {name}: gt_pass={cs['gt_pass_rate']:.4f} ({cs['gt_pass_all']}/{cs['test_outputs']}) decoy_pass={cs['decoy_pass_rate']:.3f} by_k={dict(summary[f'conj_by_k_{name}'])} failures={cs['failures'][:12]}", flush=True)
    ev = tasks[eval_key]
    models, cands, tax, flag_by_cls, fam_veto = attempt_analysis(ev, safe, hf_dir)
    summary["models"] = models
    summary["taxonomy"] = dict(tax)
    summary["flag_by_class"] = {k: {"n": v[0], "flagged": v[1]} for k, v in flag_by_cls.items()}
    summary["family_veto"] = fam_veto
    print(f"== attempts {attempt_set}: {len(models)} models, {len(cands)} attempt slots", flush=True)
    print("   taxonomy:", dict(tax))
    print("   flagged by class:", {k: f"{v[1]}/{v[0]}={v[1]/max(1,v[0]):.3f}" for k, v in flag_by_cls.items()})
    for f in FAMILIES:
        v = fam_veto[f]
        print(f"   veto {f:16s} wrong {v['wrong_vetoed']}/{v['wrong_active']}  correct {v['correct_vetoed']}/{v['correct_active']}")
    order = per_model_order(ev, models, cands)
    gains = [(m, o["pass1"], o["pass1_verifier"], o["pass2"]) for m, o in order.items()]
    per_task_delta = defaultdict(list)
    for o in order.values():
        for tid, dv in o.pop("delta_by_task").items():
            per_task_delta[tid].append(dv)
    p1_ci = bootstrap_ci([np.mean(per_task_delta.get(tid, [0.0])) for tid in ev])
    summary["per_model"] = order
    summary["pass1_gain_ci"] = p1_ci
    print(f"   pass@1 verifier ordering: mean gain={np.mean([g[2]-g[1] for g in gains]):.4f} task-bootstrap={p1_ci} models improved={sum(1 for g in gains if g[2]>g[1]+1e-9)} worse={sum(1 for g in gains if g[2]<g[1]-1e-9)} of {len(gains)}")
    print(f"   swaps={sum(o['swapped'] for o in order.values())} fixed={sum(o['fixed'] for o in order.values())} broken={sum(o['broken'] for o in order.values())}")
    for m, o in sorted(order.items(), key=lambda kv: -kv[1]["pass2"])[:8]:
        print(f"     {m:45s} tasks={o['tasks']} pass2={o['pass2']:.3f} pass1={o['pass1']:.3f} pass1_ver={o['pass1_verifier']:.3f} swaps={o['swapped']} fixed={o['fixed']} broken={o['broken']}")
    rng = random.Random(SEED)
    sel = {}
    agg_all, per_all = pooled_selection(ev, cands, models, safe, rng)
    sel["all_models"] = agg_all
    sel["all_models_ci"] = {name: bootstrap_ci([per_all[name][t] - per_all["vote"][t] for t in per_all["vote"]]) for name in ("hard", "tiebreak", "hedge")}
    for n in (2, 4, 8, 16, 32):
        vals, per_sum = [], defaultdict(lambda: defaultdict(float))
        for _ in range(200):
            agg, per = pooled_selection(ev, cands, rng.sample(models, n), safe, rng)
            vals.append(agg)
            for name, d in per.items():
                for t, v in d.items():
                    per_sum[name][t] += v / 200
        out = {name: float(np.mean([v[name] for v in vals])) for name in vals[0]}
        for name in ("hard", "tiebreak", "soft1", "hedge"):
            diffs = np.array([v[name] - v["vote"] for v in vals])
            out[name + "_wins"] = int((diffs > 1e-9).sum())
            out[name + "_losses"] = int((diffs < -1e-9).sum())
            out[name + "_ci"] = bootstrap_ci([per_sum[name][t] - per_sum["vote"][t] for t in per_sum["vote"]])
        sel[f"random_{n}"] = out
        print(f"   pooled n={n}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items() if not k.endswith("_ci")) + " CI " + " ".join(f"{k}=({v[0]:.4f},{v[1]:.4f},{v[2]:.4f})" for k, v in out.items() if k.endswith("_ci")), flush=True)
    print("   pooled all: " + " ".join(f"{k}={v:.4f}" for k, v in sel["all_models"].items()) + " CI " + str(sel["all_models_ci"]))
    summary["pooled"] = sel
    slim = [{k: c[k] for k in ("model", "task", "test", "attempt", "cls")} | {"flag": c.get("flag"), "res": c.get("res")} for c in cands]
    with open(os.path.join(OUT, f"attempts_{attempt_set}.json"), "w") as fh:
        json.dump(slim, fh)
    with open(os.path.join(OUT, "gt_rows.json"), "w") as fh:
        json.dump({k: [{kk: r[kk] for kk in ("task", "test", "k", "res")} for r in v] for k, v in rows_all.items()}, fh)
    with open(os.path.join(OUT, f"summary_{attempt_set}.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
