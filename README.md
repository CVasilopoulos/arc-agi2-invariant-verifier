# The Demonstrations Are a Verifier

Companion repository for the ARC Prize 2026 Paper Track entry "The Demonstrations Are a Verifier: Train-Pair Invariants for ARC-AGI-2 Answer Selection".

- `paper/PAPER.md` and `paper/PAPER.pdf`: the write-up, with figures in `paper/figures/`.
- `experiments/invariants.py`: the six train-pair invariants and the checker.
- `experiments/analysis.py`: the attempt-level analysis over the public Hugging Face arcprize attempt datasets (ARC-AGI-1 and ARC-AGI-2 evaluation sets).
- `experiments/nvarc_pool.py`: the same checker applied to NVARC candidate pools exported by the Kaggle notebook.
- `experiments/plots.py`: figure generation.
- `results/summary_v1.json`, `results/summary_v2.json`: analysis outputs (v2 is the one reported in the paper); `results/nvarc_pool_commit2.json`: the NVARC pool measurement.
- `notebook/`: the Kaggle notebook source and its kernel metadata. The public notebook is https://www.kaggle.com/code/chrisvas123/arc-agi-2-nvarc-invariant-verifier (ARC-AGI-2 submission 56275620, public score 29.72).

Reproduce the analysis on CPU:

```bash
pip install numpy datasets matplotlib
python experiments/analysis.py
python experiments/plots.py
```

All code and text were written with AI assistance (Claude via Claude Code) from the author's prompts and reviewed by the author; see the paper's disclosure section.

License: CC BY 4.0 (see LICENSE).
