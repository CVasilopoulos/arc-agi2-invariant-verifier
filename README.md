# The Demonstrations Are a Verifier

Companion repository for the ARC Prize 2026 Paper Track entry "The Demonstrations Are a Verifier: Train-Pair Invariants for ARC-AGI-2 Answer Selection".

- `paper/PAPER.md` and `paper/PAPER.pdf`: the write-up, with figures in `paper/figures/`.
- `paper/APPENDIX.md`: the submission configuration and the full ablation tables (family operating points, leave-one-out and add-one pooled selection, penalty size, strongest-system pools, gains and losses against the ε x pass@2 estimate, soundness on four datasets by number of demonstrations, ConceptARC per concept).
- `experiments/invariants.py`: the sixteen train-pair invariant families and the checker; the paper's verifier uses six of them.
- `experiments/analysis.py`: the attempt-level analysis over the public Hugging Face arcprize attempt datasets (ARC-AGI-1 and ARC-AGI-2 evaluation sets).
- `experiments/ablations.py`: family ablations, penalty sweep, strongest-system pools, theory check and the ConceptARC transfer measurement; `experiments/tables.py` prints the appendix tables from its output.
- `experiments/nvarc_pool.py`: the same checker applied to NVARC candidate pools exported by the Kaggle notebook.
- `experiments/plots.py`: figure generation.
- `results/summary_v1.json`, `results/summary_v2.json`: analysis outputs (v2 is the one reported in the paper); `results/ablations.json` and `results/ablations.log`: ablation outputs; `results/nvarc_pool_commit2.json`: the NVARC pool measurement.
- `notebook/`: the Kaggle notebook source and its kernel metadata. The public notebook is https://www.kaggle.com/code/chrisvas123/arc-agi-2-nvarc-invariant-verifier (ARC-AGI-2 submission 56275620, public score 29.72).
- `notebook_pools/`: a time-boxed copy of the notebook used only to export NVARC candidate pools for more evaluation tasks (not a submission).

Reproduce the analysis on CPU (ConceptARC from https://github.com/victorvikram/ConceptARC under `data/ConceptARC`):

```bash
pip install numpy datasets matplotlib
python experiments/analysis.py
python experiments/ablations.py
python experiments/tables.py
python experiments/plots.py
```

All code and text were written with AI assistance (Claude via Claude Code) from the author's prompts and reviewed by the author; see the paper's disclosure section.

License: CC BY 4.0 (see LICENSE).
