import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("ARC_ROOT", "/work")
OUT = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#d9d8d4"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def fig_soundness(s):
    fams = ["palette_subset", "palette_exact", "keep_nonbg", "input_in_output", "hist_equal", "nonbg_ge",
            "shape", "nonbg_le", "nonbg_equal", "keep_bg", "palette_keep", "sym_lr", "sym_ud", "sym_rot180", "sym_transpose"]
    safe = set(s["safe_families"])
    sets = [("agi1_eval", "ARC-AGI-1 eval", S1, "o"), ("agi2_train", "ARC-AGI-2 train (selection set)", S2, "s"), ("agi2_eval", "ARC-AGI-2 eval", S3, "D")]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ys = list(range(len(fams)))[::-1]
    for off, (key, label, color, marker) in zip((0.22, 0.0, -0.22), sets):
        xs, yy = [], []
        for f, y in zip(fams, ys):
            v = s[f"soundness_{key}"][f]
            if v["active"] >= 2:
                xs.append(v["soundness"] * 100)
                yy.append(y + off)
        ax.scatter(xs, yy, s=40, color=color, marker=marker, label=label, zorder=3, edgecolor=SURFACE, linewidth=1.0)
    labels = []
    for f in fams:
        n = s["soundness_agi2_eval"][f]["active"]
        labels.append(f"{f}{'  [safe]' if f in safe else ''}  (n={n})")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    for t, f in zip(ax.get_yticklabels(), fams):
        if f in safe:
            t.set_color(INK)
            t.set_fontweight("bold")
    ax.axvline(99, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(98.6, -0.75, "99% selection bar", color=INK2, fontsize=8, va="center", ha="right")
    ax.set_xlim(49, 101)
    ax.set_ylim(-1.2, len(fams) - 0.5)
    ax.set_xlabel("Soundness: % of true test outputs that satisfy an inferred invariant")
    ax.set_title("Invariant soundness by family", loc="left", fontsize=11)
    fig.text(0.01, 0.005, "n = test outputs on ARC-AGI-2 eval where the family is inferred from all train pairs", color=INK2, fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_soundness.png"), dpi=180)
    plt.close(fig)


def fig_attempts(s):
    order = ["correct", "near_miss", "wrong_content", "wrong_palette", "wrong_shape"]
    names = {"correct": "correct", "near_miss": "near miss (>=95% cells)", "wrong_content": "wrong content",
             "wrong_palette": "wrong palette", "wrong_shape": "wrong shape"}
    fb = s["flag_by_class"]
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    for i, c in enumerate(order[::-1]):
        n, k = fb[c]["n"], fb[c]["flagged"]
        ax.barh(i, n - k, left=k, color=NEUTRAL, height=0.62, edgecolor=SURFACE, linewidth=2)
        ax.barh(i, k, color=S1, height=0.62, edgecolor=SURFACE, linewidth=2)
        ax.text(n + 90, i, f"{100 * k / n:.1f}% of {n:,}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[c] for c in order[::-1]])
    ax.set_xlim(0, 10500)
    ax.set_xlabel("Valid attempts from 68 systems on ARC-AGI-2 eval (blue = flagged)")
    fig.suptitle("Which attempts the verifier flags", x=0.01, ha="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_attempts.png"), dpi=180)
    plt.close(fig)


def fig_pooled(s, path, title):
    p = s["pooled"]
    ns = [2, 4, 8, 16, 32]
    series = [("tiebreak", "verifier breaks vote ties", S1, "o"), ("hard", "verifier vetoes (hard)", S2, "s"), ("hedge", "verifier picks 2nd attempt", S3, "D")]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0, color=INK2, linewidth=1)
    for key, label, color, marker in series:
        ys = [(p[f"random_{n}"][key] - p[f"random_{n}"]["vote"]) * 100 for n in ns]
        ys.append((p["all_models"][key] - p["all_models"]["vote"]) * 100)
        ax.plot(ns + [68], ys, color=color, linewidth=2, marker=marker, markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns + [68])
    ax.set_xticklabels([str(n) for n in ns] + ["all 68"])
    ax.set_xlim(1.7, 85)
    ax.set_xlabel("Systems pooled (random subsets, 200 draws; last point = all 68)")
    ax.set_ylabel("Pass@2 gain over majority vote, points")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_operating_points(a):
    att = a["attempts"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    shown = {"palette_keep": "palette_keep", "shape": "shape", "nonbg_le": "nonbg_le", "nonbg_ge": "nonbg_ge", "palette_exact": "palette_exact",
             "nonbg_equal": "nonbg_equal", "keep_nonbg": "keep_nonbg", "sym_ud": "sym_ud"}
    for f, v in att["families"].items():
        x, y = v["false_flag"] * 100, v["recall"] * 100
        ax.scatter(x, y, s=28, color=NEUTRAL, edgecolor=INK2, linewidth=0.8, zorder=3)
        if f in shown:
            ax.annotate(shown[f], (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    sets = [("safe6", "safe six (paper)", S1, "D"), ("safe6+keep_bg", "+ keep_bg", S3, "o"), ("safe6+nonbg_equal", "+ nonbg_equal", S3, "s"),
            ("safe6+nonbg_le", "+ nonbg_le", S3, "^"), ("safe6+shape", "+ shape", S2, "v"), ("all16", "all sixteen", S2, "X")]
    for key, label, color, marker in sets:
        v = att["sets"][key]
        ax.scatter(v["false_flag"] * 100, v["recall"] * 100, s=70, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.0, zorder=4, label=label)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlim(-0.02, 40)
    ax.set_xticks([0, 0.1, 0.3, 1, 3, 10, 30])
    ax.set_xticklabels(["0", "0.1", "0.3", "1", "3", "10", "30"])
    ax.set_ylim(0, 50)
    ax.set_xlabel("Correct attempts flagged, % (false flags)")
    ax.set_ylabel("Wrong attempts flagged, % (recall)")
    ax.set_title("Verifier operating points on 21,274 valid ARC-AGI-2 attempts", loc="left", fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.text(0.01, 0.005, "grey: single families; coloured: conjunctions of the safe six with one more family", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_operating_points.png"), dpi=180)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(OUT, "summary_v2.json")) as fh:
        s = json.load(fh)
    fig_soundness(s)
    fig_attempts(s)
    fig_pooled(s, os.path.join(FIG, "fig3_pooled.png"), "Adding the verifier to majority-vote selection (ARC-AGI-2 eval)")
    fig_pooled(s, os.path.join(FIG, "cover.png"), "Train-pair invariants as a free verifier on ARC-AGI-2")
    abl = os.path.join(OUT, "ablations.json")
    if os.path.exists(abl):
        with open(abl) as fh:
            fig_operating_points(json.load(fh))


if __name__ == "__main__":
    main()
