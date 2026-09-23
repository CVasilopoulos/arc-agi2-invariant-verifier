import json
import os

ROOT = os.environ.get("ARC_ROOT", "/work")
OUT = os.path.join(ROOT, "results")
SIZES = ["random_2", "random_4", "random_8", "random_16", "random_32", "all_models"]
SIZE_NAMES = ["2", "4", "8", "16", "32", "all 68"]
SET_ORDER = ["safe6", "safe6-palette_exact", "safe6-palette_subset", "safe6-hist_equal", "safe6-nonbg_ge", "safe6-keep_nonbg", "safe6-input_in_output",
             "safe6+keep_bg", "safe6+nonbg_equal", "safe6+nonbg_le", "safe6+shape", "safe6+sym_lr", "safe6+palette_keep", "all16", "shape_only"]


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    with open(os.path.join(OUT, "ablations.json")) as fh:
        a = json.load(fh)
    att, snd, pooled, theory = a["attempts"], a["soundness"], a["pooled"], a["theory"]
    print("### A1. Operating points of single families on ARC-AGI-2 attempts\n")
    rows = [(f, pct(v["active_share"]), pct(v["recall"]), pct(v["false_flag"], 2)) for f, v in att["families"].items()]
    print(table(["Family", "Inferred on % of attempts", "Wrong flagged %", "Correct flagged %"], rows))
    print("\n### A2. Operating points of family sets\n")
    rows = [(s, pct(att["sets"][s]["recall"]), pct(att["sets"][s]["false_flag"], 2)) + tuple(pct(att["by_class"][s][c]) for c in ("near_miss", "wrong_content", "wrong_palette", "wrong_shape")) for s in SET_ORDER]
    print(table(["Set", "Wrong flagged %", "Correct flagged %", "Near miss %", "Wrong content %", "Wrong palette %", "Wrong shape %"], rows))
    for lam, title in (("lam1", "A3. Pooled pass@2 gain over vote, one-vote penalty (points; wins/losses over draws)"), ("hard", "A4. Pooled pass@2 gain over vote, hard veto")):
        print(f"\n### {title}\n")
        rows = []
        for s in SET_ORDER:
            r = [s]
            for size in SIZES:
                v = pooled[size][f"{s}|{lam}"]
                r.append(f"{100 * v['gain']:+.2f} ({v['wins']}/{v['losses']})")
            rows.append(r)
        print(table(["Set"] + SIZE_NAMES, rows))
    print("\n### A5. Penalty size for the safe six\n")
    rows = []
    for name in ("safe6|lam0.5", "safe6|lam1", "safe6|lam2.0", "safe6|hard"):
        r = [name.split("|")[1]]
        for size in SIZES:
            v = pooled[size][name]
            ci = f" [{100 * v['ci'][1]:+.1f}, {100 * v['ci'][2]:+.1f}]" if size != "all_models" else ""
            r.append(f"{100 * v['gain']:+.2f}{ci}")
        rows.append(r)
    print(table(["Penalty"] + SIZE_NAMES, rows))
    print("\n### A6. Strongest systems pooled (top-N by pass@2; 20 tie-break draws)\n")
    rows = []
    for n in (2, 4, 8, 16):
        v = pooled[f"top_{n}"]
        rows.append((n, pct(v["vote"]["pass2"]), pct(v["oracle"]["pass2"])) + tuple(f"{100 * v[k]['gain']:+.2f} [{100 * v[k]['ci'][1]:+.1f}, {100 * v[k]['ci'][2]:+.1f}]" for k in ("safe6|lam1", "safe6|hard", "all16|lam1", "all16|hard")))
    print(table(["Top N", "Vote", "Oracle", "Safe six, one vote", "Safe six, hard", "All 16, one vote", "All 16, hard"], rows))
    print("\nTop 16 by pass@2: " + ", ".join(pooled["top_16"]["models"]))
    print("\n### A7. Gains and losses per output vs the bound eps x pass@2 (% of 167 outputs)\n")
    rows = []
    for size, name in zip(SIZES, SIZE_NAMES):
        t = theory[size]
        rows.append((name, pct(t["vote_pass2"]), pct(t["predicted_loss_bound"], 2), pct(t["hard_gain"], 2), pct(t["hard_loss"], 2), pct(t["lam1_gain"], 2), pct(t["lam1_loss"], 2), pct(t["gap_closed_lam1"])))
    print(table(["Systems", "Vote pass@2", "eps x pass@2", "Hard gain", "Hard loss", "One-vote gain", "One-vote loss", "Gap to oracle closed"], rows))
    print("\n### A8. Soundness of family sets on true test outputs (pass %) and decoy pass %\n")
    rows = []
    for s in SET_ORDER:
        r = [s]
        for ds in ("agi2_train", "agi2_eval", "agi1_eval", "conceptarc"):
            v = snd[ds]["sets"][s]
            r.append(f"{pct(v['gt_pass_rate'])} / {pct(v['decoy_pass_rate'])}")
        rows.append(r)
    print(table(["Set", "ARC-AGI-2 train (1,076)", "ARC-AGI-2 eval (167)", "ARC-AGI-1 eval (419)", "ConceptARC (480)"], rows))
    print("\n### A9. Safe six by number of demonstrations k: true outputs failing % (n) and decoy pass %\n")
    rows = []
    for ds, name in (("agi2_train", "ARC-AGI-2 train"), ("agi2_eval", "ARC-AGI-2 eval"), ("agi1_eval", "ARC-AGI-1 eval"), ("conceptarc", "ConceptARC")):
        r = [name]
        for k in ("2", "3", "4", "5"):
            v = snd[ds]["by_k"]["safe6"].get(k)
            r.append("-" if not v else f"{pct(v['gt_fail'] / v['n'])} ({v['n']}) / {pct(v['decoy_pass'] / max(1, v['decoys']))}")
        rows.append(r)
    print(table(["Dataset", "k=2", "k=3", "k=4", "k>=5"], rows))
    print("\n### A10. Safe six recall and false flags on ARC-AGI-2 attempts by k\n")
    rows = []
    for k, v in att["by_k"]["safe6"].items():
        rows.append((k if k != "5" else ">=5", v["wrong"], pct(v["wrong_flagged"] / v["wrong"]), v["correct"], pct(v["correct_flagged"] / v["correct"], 2)))
    print(table(["k", "Wrong attempts", "Flagged %", "Correct attempts", "Flagged %"], rows))
    print("\n### A11. ConceptARC per concept (safe six, true outputs passing)\n")
    rows = [(c, f"{v['pass']}/{v['n']}") for c, v in snd["conceptarc"]["per_concept"].items()]
    print(table(["Concept", "Pass"], rows))
    fails = snd["conceptarc"]["sets"]["safe6"]["failures"]
    print("\nFailures: " + "; ".join(f"{t} test {i} ({', '.join(fs)})" for t, i, fs in fails))


if __name__ == "__main__":
    main()
