import bz2
import json
import os
import pickle
import tarfile
from collections import Counter

import numpy as np

from analysis import DATASETS, classify, load_dir
from invariants import TaskInvariants, as_grid

ROOT = os.environ.get("ARC_ROOT", "/work")
POOL = os.environ.get("POOL_TAR", os.path.join(ROOT, "runs", "commit2", "nvarc_candidate_pools.tar.gz"))
OUT = os.environ.get("POOL_OUT", os.path.join(ROOT, "results", "nvarc_pool_commit2.json"))
SAFE = ["palette_exact", "palette_subset", "hist_equal", "nonbg_ge", "keep_nonbg", "input_in_output"]
LAMBDAS = [0.0, 0.5, 1.0, 2.0, float("inf")]


def load_pool(path):
    pools = {}
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            base = name.split(".")[0]
            samples = pickle.loads(bz2.decompress(tar.extractfile(m).read()))
            for i, s in enumerate(samples):
                pools.setdefault(base, {})[f"{name}.out{i}"] = s
    return pools


def distinct_candidates(guesses):
    groups = {}
    for g in guesses.values():
        sol = np.asarray(g["solution"], dtype=np.int16)
        key = tuple(map(tuple, sol.tolist()))
        x = groups.setdefault(key, {"solution": sol, "hits": 0, "aug": [], "beam": []})
        x["hits"] += 1
        x["aug"].append(float(np.mean(g["score_aug"])))
        x["beam"].append(float(g["beam_score"]))
    out = list(groups.values())
    for x in out:
        x["kgmon"] = x["hits"] - float(np.mean(x["aug"]))
    return sorted(out, key=lambda c: c["kgmon"], reverse=True)


def order(cands, lam):
    def score(c):
        if not c["flag"]:
            return c["kgmon"]
        return float("-inf") if lam == float("inf") else c["kgmon"] - lam
    return sorted(cands, key=score, reverse=True)


def top2_correct(cands):
    return any(c["correct"] for c in cands[:2])


def main():
    tasks = load_dir(DATASETS["agi2_eval"])
    pools = load_pool(POOL)
    rows = []
    skipped = [b for b in pools if int(b.split("_")[1]) >= len(tasks[b.split("_")[0]]["test"])]
    for b in skipped:
        del pools[b]
    per_task_outputs = Counter(k.split("_")[0] for k in pools)
    totals = Counter()
    for base in sorted(pools):
        tid, ti = base.split("_")
        task = tasks[tid]
        test = task["test"][int(ti)]
        gt = as_grid(test["output"])
        inv = TaskInvariants([(p["input"], p["output"]) for p in task["train"]])
        cands = distinct_candidates(pools[base])
        for rank, c in enumerate(cands, 1):
            res = inv.check(test["input"], c["solution"])
            c["violated"] = [f for f in SAFE if not res.get(f, True)]
            c["flag"] = bool(c["violated"])
            c["class"] = classify(c["solution"], gt)
            c["correct"] = c["class"] == "correct"
            c["baseline_rank"] = rank
        base_top2 = [c["baseline_rank"] for c in cands[:2]]
        row = {
            "output": base,
            "beams": len(pools[base]),
            "distinct": len(cands),
            "classes": dict(Counter(c["class"] for c in cands)),
            "correct_in_pool": any(c["correct"] for c in cands),
            "correct_baseline_rank": next((c["baseline_rank"] for c in cands if c["correct"]), None),
            "kgmon_top3": [round(c["kgmon"], 3) for c in cands[:3]],
            "flagged": [{"baseline_rank": c["baseline_rank"], "class": c["class"], "hits": c["hits"],
                         "kgmon": round(c["kgmon"], 3), "violated": c["violated"]} for c in cands if c["flag"]],
            "selection": {},
        }
        for lam in LAMBDAS:
            o = order(cands, lam)
            row["selection"][str(lam)] = {
                "top2_correct": top2_correct(o),
                "top2_changed": [c["baseline_rank"] for c in o[:2]] != base_top2,
                "top2_baseline_ranks": [c["baseline_rank"] for c in o[:2]],
            }
        rows.append(row)
        for c in cands:
            totals["distinct"] += 1
            totals["wrong"] += not c["correct"]
            totals["flagged"] += c["flag"]
            totals["flagged_wrong"] += c["flag"] and not c["correct"]
            totals["flagged_correct"] += c["flag"] and c["correct"]
    scores = {}
    for lam in LAMBDAS:
        s = 0.0
        for r in rows:
            if r["selection"][str(lam)]["top2_correct"]:
                s += 1 / per_task_outputs[r["output"].split("_")[0]]
        scores[str(lam)] = {"score": round(s, 3), "tasks": len(per_task_outputs),
                            "outputs_correct": sum(r["selection"][str(lam)]["top2_correct"] for r in rows),
                            "outputs_top2_changed": sum(r["selection"][str(lam)]["top2_changed"] for r in rows)}
    summary = {"pool": os.path.basename(os.path.dirname(POOL)), "outputs": len(rows), "tasks": len(per_task_outputs),
               "skipped_no_ground_truth": sorted(skipped),
               "totals": dict(totals), "scores": scores, "rows": rows}
    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"pool {summary['pool']}: {summary['tasks']} tasks, {summary['outputs']} test outputs, "
          f"{sum(r['beams'] for r in rows)} beams, {totals['distinct']} distinct candidates "
          f"({totals['wrong']} wrong)")
    if skipped:
        print(f"skipped {len(skipped)} outputs with no ground truth in the public copy: {sorted(skipped)}")
    print(f"flagged {totals['flagged']}: {totals['flagged_wrong']} wrong, {totals['flagged_correct']} correct")
    print(f"{'output':12s} {'beams':>5s} {'dist':>4s} {'c_in':>4s} {'c_rank':>6s} {'flag':>4s} {'base':>5s} {'l=1':>5s} {'hard':>5s} {'chg1':>5s} kgmon_top3 classes")
    for r in rows:
        s = r["selection"]
        print(f"{r['output']:12s} {r['beams']:5d} {r['distinct']:4d} {str(r['correct_in_pool']):>4s} "
              f"{str(r['correct_baseline_rank']):>6s} {len(r['flagged']):4d} {str(s['0.0']['top2_correct']):>5s} "
              f"{str(s['1.0']['top2_correct']):>5s} {str(s['inf']['top2_correct']):>5s} "
              f"{str(s['1.0']['top2_changed']):>5s} {r['kgmon_top3']} {r['classes']}")
        for f in r["flagged"]:
            print(f"    flagged rank {f['baseline_rank']} {f['class']} hits={f['hits']} kgmon={f['kgmon']} violated={f['violated']}")
    for lam, v in scores.items():
        print(f"lambda={lam:>4s}: task score {v['score']}/{v['tasks']}, outputs correct {v['outputs_correct']}/{len(rows)}, top-2 changed {v['outputs_top2_changed']}")


if __name__ == "__main__":
    main()
