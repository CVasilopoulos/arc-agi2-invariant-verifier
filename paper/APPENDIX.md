# Appendix: The Demonstrations Are a Verifier

Companion tables for the ARC Prize 2026 Paper Track write-up. All numbers come from `results/ablations.json` (produced by `experiments/ablations.py`, seed 20260916, 200 random pools per size) and `results/summary_v2.json` / `results/summary_v1.json` (`experiments/analysis.py`); the tables below are printed by `experiments/tables.py`. Attempts: Hugging Face `arcprize/arc_agi_v2_public_eval` (68 systems with at least 90 tasks) and `arcprize/arc_agi_v1_public_eval` (73 systems). "Wrong flagged %" is recall over all valid wrong attempts, not only those where the family is inferred; "Correct flagged %" is the false-flag rate over all correct attempts.

## A0. The Kaggle submission

Notebook `chrisvas123/arc-agi-2-nvarc-invariant-verifier`, version 2, submission 56275620, public score 29.72. Fork of the NVARC 2025 winning notebook as republished for 2026 (`yiheng/reproduce-nvarc-2025-results`); solver, test-time training, decoding and scoring unchanged.

| Component | Setting |
|---|---|
| Model | `sorokin/qwen3_4b_grids15_sft139` (Qwen3-4B fine-tuned on synthetic ARC-style tasks), bf16, max sequence 8,192 tokens |
| Test-time training | per task, LoRA r=256, alpha 32, rsLoRA, on q/k/v/o/gate/up/down projections plus embed_tokens and lm_head; one epoch over 16 augmented copies of the demonstration pairs (transpose, rotations, colour permutations, shuffled pair order); lr 5e-5, cosine, AdamW, batch 1 |
| Decoding | depth-first search on 16 augmented views of the test input (8 geometric views x 2 colour permutations), keeping every completion whose cumulative probability stays above 0.2; DFS capped at 540 s per batch and 1,200 s per task |
| Candidate score (kgmon) | number of augmented views that decoded the candidate minus its mean negative log-likelihood over 8 augmented views |
| Verifier | six invariant families inferred from the task's train pairs; a candidate violating any inferred family loses `FLAG_PENALTY = 1.0` from kgmon; families and λ fixed before the run |
| Hardware and budget | 4 x NVIDIA L4 (one worker each), 12-hour budget minus 10 minutes; the commit run on four public evaluation tasks took 1,584 s |
| Outputs | `submission.json`, `verifier_log.json` (candidates, flags, top-2 changes per output), `nvarc_candidate_pools.tar.gz` (all decoded beams with scores, commit mode only) |

Commit-mode result on the four evaluation tasks: baseline kgmon 3.0/4, verified selection 3.0/4; one output (36a08778_1) had one flagged candidate and a changed top-2.

### A1. Operating points of single families on ARC-AGI-2 attempts

| Family | Inferred on % of attempts | Wrong flagged % | Correct flagged % |
|---|---|---|---|
| shape | 76.8 | 9.1 | 4.20 |
| palette_exact | 69.5 | 9.5 | 0.05 |
| palette_subset | 100.0 | 1.3 | 0.00 |
| palette_keep | 100.0 | 21.8 | 17.56 |
| hist_equal | 5.9 | 4.1 | 0.21 |
| nonbg_equal | 17.3 | 9.7 | 0.21 |
| nonbg_ge | 53.5 | 9.2 | 0.73 |
| nonbg_le | 54.9 | 5.0 | 0.98 |
| keep_nonbg | 17.3 | 6.1 | 0.00 |
| keep_bg | 10.2 | 1.9 | 0.00 |
| crop_of_input | 0.0 | 0.0 | 0.00 |
| input_in_output | 1.2 | 0.5 | 0.00 |
| sym_lr | 4.1 | 2.0 | 0.82 |
| sym_ud | 3.4 | 2.1 | 1.46 |
| sym_rot180 | 1.8 | 1.0 | 0.00 |
| sym_transpose | 1.8 | 1.2 | 0.53 |

### A2. Operating points of family sets

`safe6` is the paper's verifier; `safe6-x` removes one family, `safe6+x` adds one, `all16` uses every family.

| Set | Wrong flagged % | Correct flagged % | Near miss % | Wrong content % | Wrong palette % | Wrong shape % |
|---|---|---|---|---|---|---|
| safe6 | 24.4 | 0.78 | 14.3 | 18.0 | 44.1 | 28.7 |
| safe6-palette_exact | 17.6 | 0.73 | 14.3 | 17.1 | 18.0 | 20.1 |
| safe6-palette_subset | 24.0 | 0.78 | 14.3 | 18.0 | 43.2 | 27.6 |
| safe6-hist_equal | 23.2 | 0.78 | 12.8 | 16.4 | 44.1 | 27.8 |
| safe6-nonbg_ge | 19.5 | 0.25 | 10.0 | 14.8 | 38.7 | 21.1 |
| safe6-keep_nonbg | 19.1 | 0.78 | 12.5 | 9.4 | 42.3 | 24.3 |
| safe6-input_in_output | 23.9 | 0.78 | 14.3 | 17.0 | 44.1 | 28.3 |
| safe6+keep_bg | 25.7 | 0.78 | 16.3 | 19.6 | 44.5 | 29.6 |
| safe6+nonbg_equal | 25.8 | 0.78 | 16.0 | 19.5 | 44.1 | 30.7 |
| safe6+nonbg_le | 27.8 | 1.76 | 18.3 | 21.6 | 45.5 | 32.7 |
| safe6+shape | 29.5 | 4.98 | 15.3 | 20.2 | 47.7 | 42.2 |
| safe6+sym_lr | 26.0 | 1.60 | 14.7 | 19.1 | 44.4 | 32.7 |
| safe6+palette_keep | 36.7 | 18.29 | 19.7 | 23.5 | 69.4 | 48.3 |
| all16 | 45.5 | 24.27 | 27.4 | 30.0 | 72.7 | 65.1 |
| shape_only | 9.1 | 4.20 | 1.0 | 2.2 | 4.1 | 29.1 |

### A3. Pooled pass@2 gain over vote, one-vote penalty (points; wins/losses over 200 draws)

These draws differ from those of the paper's main table (a separate random sequence); vote baselines: 30.5, 41.1, 56.1, 71.9, 80.3, 84.9.

| Set | 2 | 4 | 8 | 16 | 32 | all 68 |
|---|---|---|---|---|---|---|
| safe6 | +0.85 (130/5) | +3.67 (188/1) | +1.91 (179/1) | +0.76 (129/7) | +0.33 (70/1) | +0.00 (0/0) |
| safe6-palette_exact | +0.59 (108/13) | +2.64 (176/3) | +1.39 (163/3) | +0.46 (100/9) | +0.14 (37/2) | +0.00 (0/0) |
| safe6-palette_subset | +0.83 (128/5) | +3.64 (188/1) | +1.87 (179/1) | +0.76 (129/7) | +0.33 (70/1) | +0.00 (0/0) |
| safe6-hist_equal | +0.78 (127/5) | +3.43 (186/1) | +1.73 (175/1) | +0.68 (121/8) | +0.30 (63/1) | +0.00 (0/0) |
| safe6-nonbg_ge | +0.75 (131/1) | +2.78 (183/0) | +1.39 (168/0) | +0.57 (113/3) | +0.25 (56/1) | +0.00 (0/0) |
| safe6-keep_nonbg | +0.61 (113/6) | +2.79 (183/3) | +1.47 (162/2) | +0.66 (114/7) | +0.31 (69/1) | +0.00 (0/0) |
| safe6-input_in_output | +0.85 (130/5) | +3.67 (188/1) | +1.91 (179/1) | +0.76 (129/7) | +0.33 (70/1) | +0.00 (0/0) |
| safe6+keep_bg | +1.01 (138/4) | +3.99 (190/1) | +2.11 (184/1) | +0.77 (131/7) | +0.34 (72/1) | +0.00 (0/0) |
| safe6+nonbg_equal | +0.99 (136/4) | +4.02 (190/0) | +2.23 (185/1) | +0.82 (133/7) | +0.36 (72/1) | +0.00 (0/0) |
| safe6+nonbg_le | +0.88 (133/15) | +4.06 (189/3) | +2.30 (182/3) | +0.83 (135/11) | +0.37 (76/1) | +0.00 (0/0) |
| safe6+shape | +0.88 (131/17) | +3.85 (189/3) | +1.92 (178/6) | +0.71 (128/16) | +0.31 (69/5) | +0.00 (0/0) |
| safe6+sym_lr | +0.88 (134/4) | +3.86 (189/0) | +1.98 (179/2) | +0.75 (128/7) | +0.33 (70/1) | +0.00 (0/0) |
| safe6+palette_keep | +0.43 (101/39) | +2.72 (176/11) | +1.49 (162/17) | +0.61 (121/21) | +0.33 (78/9) | +0.00 (0/0) |
| all16 | +0.62 (123/37) | +3.26 (182/8) | +1.78 (167/15) | +0.60 (125/30) | +0.30 (73/11) | +0.00 (0/0) |
| shape_only | +0.23 (52/23) | +0.72 (105/21) | +0.20 (73/27) | -0.04 (19/23) | -0.00 (9/7) | +0.00 (0/0) |

### A4. Pooled pass@2 gain over vote, hard veto

| Set | 2 | 4 | 8 | 16 | 32 | all 68 |
|---|---|---|---|---|---|---|
| safe6 | +0.86 (131/5) | +3.56 (179/10) | +1.95 (159/25) | +0.97 (135/43) | +0.92 (126/46) | -0.75 (0/18) |
| safe6-palette_exact | +0.59 (109/14) | +2.47 (167/14) | +1.21 (134/42) | +0.36 (106/62) | +0.11 (76/77) | -0.75 (0/18) |
| safe6-palette_subset | +0.84 (129/5) | +3.53 (179/10) | +1.91 (158/25) | +0.96 (135/43) | +0.92 (126/46) | -0.75 (0/18) |
| safe6-hist_equal | +0.79 (128/5) | +3.34 (178/10) | +1.74 (152/30) | +0.84 (128/46) | +0.85 (124/48) | -0.75 (0/18) |
| safe6-nonbg_ge | +0.75 (130/1) | +2.84 (183/2) | +1.67 (169/4) | +0.92 (143/19) | +1.01 (147/15) | +0.00 (0/0) |
| safe6-keep_nonbg | +0.61 (115/6) | +2.67 (176/11) | +1.51 (147/32) | +0.85 (127/50) | +0.89 (125/46) | -0.75 (0/18) |
| safe6-input_in_output | +0.86 (131/5) | +3.56 (179/10) | +1.95 (159/25) | +0.97 (135/43) | +0.92 (126/46) | -0.75 (0/18) |
| safe6+keep_bg | +1.01 (139/4) | +3.89 (184/7) | +2.17 (164/20) | +1.00 (137/42) | +0.93 (126/46) | -0.75 (0/18) |
| safe6+nonbg_equal | +0.99 (136/4) | +3.93 (185/5) | +2.32 (168/18) | +1.09 (138/42) | +1.02 (128/42) | -0.75 (0/18) |
| safe6+nonbg_le | +0.87 (133/16) | +3.74 (179/12) | +1.82 (137/46) | +0.13 (91/93) | -0.17 (78/110) | -2.00 (0/20) |
| safe6+shape | +0.89 (131/17) | +3.59 (178/12) | +1.34 (130/50) | -0.95 (46/144) | -2.14 (14/176) | -4.08 (0/20) |
| safe6+sym_lr | +0.89 (135/4) | +3.75 (182/8) | +2.01 (156/24) | +0.91 (134/45) | +0.75 (119/58) | -1.17 (0/20) |
| safe6+palette_keep | +0.43 (102/39) | +1.93 (143/45) | -1.84 (42/155) | -6.92 (0/200) | -10.35 (0/200) | -13.53 (0/20) |
| all16 | +0.62 (123/37) | +2.31 (146/42) | -1.67 (46/149) | -7.12 (1/199) | -11.24 (0/200) | -14.36 (0/20) |
| shape_only | +0.23 (52/23) | +0.55 (89/42) | -0.48 (37/120) | -1.95 (1/196) | -3.06 (0/200) | -3.33 (0/20) |

### A5. Penalty size for the safe six (gain in points, 95% task bootstrap)

| Penalty | 2 | 4 | 8 | 16 | 32 | all 68 |
|---|---|---|---|---|---|---|
| λ=0.5 | +0.85 [+0.5, +1.2] | +3.67 [+2.9, +4.5] | +1.91 [+1.4, +2.4] | +0.76 [+0.4, +1.1] | +0.33 [+0.1, +0.6] | +0.00 |
| λ=1 | +0.85 [+0.5, +1.2] | +3.67 [+2.9, +4.5] | +1.91 [+1.4, +2.4] | +0.76 [+0.4, +1.1] | +0.33 [+0.1, +0.6] | +0.00 |
| λ=2 | +0.86 [+0.5, +1.2] | +3.59 [+2.6, +4.6] | +2.00 [+1.1, +2.8] | +0.98 [+0.3, +1.6] | +0.76 [+0.1, +1.5] | +0.00 |
| hard veto | +0.86 [+0.5, +1.2] | +3.56 [+2.5, +4.6] | +1.95 [+0.6, +3.0] | +0.97 [-0.5, +2.1] | +0.92 [-0.7, +2.4] | -0.75 |

### A6. Strongest systems pooled (top-N by pass@2; 20 tie-break draws)

| Top N | Vote | Oracle | Safe six, one vote | Safe six, hard | All 16, one vote | All 16, hard |
|---|---|---|---|---|---|---|
| 2 | 90.8 | 92.5 | +0.00 [+0.0, +0.0] | +0.00 [+0.0, +0.0] | +0.00 [+0.0, +0.0] | +0.00 [+0.0, +0.0] |
| 4 | 91.9 | 93.3 | +0.17 [+0.0, +0.4] | -0.25 [-1.3, +0.4] | +0.17 [+0.0, +0.4] | +0.21 [-1.1, +1.5] |
| 8 | 89.7 | 97.5 | +1.06 [+0.0, +2.8] | +0.65 [-0.9, +2.6] | +1.06 [+0.0, +2.8] | +0.37 [-2.5, +3.4] |
| 16 | 88.1 | 97.5 | +0.25 [+0.0, +0.6] | -0.71 [-3.0, +1.1] | +0.14 [-0.2, +0.6] | -2.07 [-5.6, +1.3] |

Top 16 by pass@2: gpt-5-4-pro-xhigh, gemini-3-1-pro-preview, claude-opus-4-8-max, claude-opus-4-6-thinking-120K-high, claude-opus-4-8-high, claude-opus-4-6-thinking-120K-max, gpt-5-4-high, claude-opus-4-8-medium, claude-opus-4-6-thinking-120K-medium, claude-opus-4-8-low, grok-4.20-beta-0309b-reasoning, grok-4.20-multi-agent-beta-0309-xhigh, claude-opus-4-6-thinking-120K-low, gpt-5-2-2025-12-11-thinking-xhigh, gpt-5-4-medium, gpt-5-2-pro-2025-12-11-high.

### A7. Gains and losses per output against the independence estimate ε x pass@2 (% of 167 outputs; ε = 3/167)

| Systems | Vote pass@2 | ε x pass@2 | Hard gain | Hard loss | One-vote gain | One-vote loss | Gap to oracle closed (one vote) |
|---|---|---|---|---|---|---|---|
| 2 | 30.5 | 0.55 | 0.99 | 0.09 | 0.99 | 0.09 | 16.8 |
| 4 | 41.1 | 0.74 | 3.81 | 0.28 | 3.69 | 0.09 | 19.6 |
| 8 | 56.1 | 1.01 | 2.44 | 0.45 | 2.02 | 0.07 | 9.1 |
| 16 | 71.9 | 1.29 | 1.40 | 0.56 | 0.74 | 0.03 | 4.9 |
| 32 | 80.3 | 1.44 | 1.23 | 0.54 | 0.28 | 0.01 | 2.5 |
| all 68 | 84.9 | 1.52 | 0.06 | 0.60 | 0.00 | 0.00 | 0.0 |

The observed hard-veto loss stays below ε x pass@2 because a loss also needs at least two unflagged candidates below the vetoed correct answer; on the full pool only one of the three coverage-gap outputs meets that condition.

### A8. Soundness of family sets on true test outputs (pass %) / decoy pass % (20 outputs of unrelated tasks per test)

| Set | ARC-AGI-2 train (1,076) | ARC-AGI-2 eval (167) | ARC-AGI-1 eval (419) | ConceptARC (480) |
|---|---|---|---|---|
| safe6 | 99.2 / 4.1 | 98.2 / 9.4 | 99.8 / 4.5 | 95.0 / 8.6 |
| safe6-palette_exact | 99.5 / 8.8 | 98.8 / 15.9 | 100.0 / 11.7 | 97.7 / 12.3 |
| safe6-palette_subset | 99.3 / 14.7 | 98.2 / 27.9 | 99.8 / 12.8 | 95.6 / 32.5 |
| safe6-hist_equal | 99.2 / 4.1 | 98.2 / 9.4 | 99.8 / 4.6 | 95.0 / 8.8 |
| safe6-nonbg_ge | 99.4 / 4.8 | 98.8 / 10.1 | 99.8 / 5.7 | 95.4 / 9.3 |
| safe6-keep_nonbg | 99.3 / 4.4 | 98.2 / 9.8 | 99.8 / 4.9 | 95.6 / 9.0 |
| safe6-input_in_output | 99.2 / 4.2 | 98.2 / 9.4 | 99.8 / 4.6 | 95.0 / 8.6 |
| safe6+keep_bg | 98.5 / 3.7 | 98.2 / 8.7 | 98.6 / 4.2 | 94.6 / 3.4 |
| safe6+nonbg_equal | 98.6 / 3.9 | 98.2 / 8.5 | 99.0 / 4.4 | 94.6 / 8.5 |
| safe6+nonbg_le | 98.4 / 2.6 | 97.0 / 5.7 | 98.6 / 3.2 | 94.0 / 6.9 |
| safe6+shape | 98.0 / 1.3 | 94.0 / 5.7 | 99.5 / 2.3 | 91.5 / 2.4 |
| safe6+sym_lr | 99.1 / 4.0 | 97.0 / 9.0 | 99.8 / 4.5 | 92.7 / 8.5 |
| safe6+palette_keep | 92.4 / 1.4 | 82.6 / 1.2 | 95.5 / 1.6 | 70.8 / 2.4 |
| all16 | 90.1 / 0.2 | 75.4 / 0.4 | 93.1 / 0.1 | 63.3 / 0.2 |
| shape_only | 98.7 / 13.2 | 95.8 / 25.9 | 99.8 / 12.4 | 96.2 / 20.8 |

### A9. Safe six by number of demonstrations k: true outputs failing % (n) / decoy pass %

| Dataset | k=2 | k=3 | k=4 | k>=5 |
|---|---|---|---|---|
| ARC-AGI-2 train | 1.2 (164) / 1.8 | 1.0 (606) / 4.6 | 0.5 (205) / 3.5 | 0.0 (101) / 6.2 |
| ARC-AGI-2 eval | 2.1 (48) / 10.8 | 2.5 (80) / 8.5 | 0.0 (26) / 13.1 | 0.0 (13) / 2.3 |
| ARC-AGI-1 eval | 0.0 (45) / 2.4 | 0.5 (221) / 4.6 | 0.0 (92) / 2.8 | 0.0 (61) / 8.5 |
| ConceptARC | 7.6 (225) / 6.1 | 2.7 (183) / 12.3 | 3.0 (66) / 7.1 | 0.0 (3) / 8.3 |

Failures fall with k (coverage gaps close) while decoy pass rates rise with k (fewer families survive inference).

### A10. Safe six recall and false flags on ARC-AGI-2 attempts by k

| k | Wrong attempts | Flagged % | Correct attempts | Flagged % |
|---|---|---|---|---|
| 2 | 4791 | 30.7 | 1241 | 0.16 |
| 3 | 8059 | 23.8 | 2200 | 1.45 |
| 4 | 2701 | 12.9 | 607 | 0.00 |
| >=5 | 1344 | 28.3 | 331 | 0.00 |

### A11. ConceptARC (Moskvichev, Odouard and Mitchell 2023), safe six, true outputs passing per concept

| Concept | Pass |
|---|---|
| AboveBelow | 25/30 |
| Center | 30/30 |
| CleanUp | 28/30 |
| CompleteShape | 26/30 |
| Copy | 29/30 |
| Count | 29/30 |
| ExtendToBoundary | 30/30 |
| ExtractObjects | 28/30 |
| FilledNotFilled | 29/30 |
| HorizontalVertical | 27/30 |
| InsideOutside | 29/30 |
| MoveToBoundary | 29/30 |
| Order | 28/30 |
| SameDifferent | 30/30 |
| TopBottom2D | 29/30 |
| TopBottom3D | 30/30 |

Failures by family: palette_exact 14, keep_nonbg 5, nonbg_ge 4, palette_subset 4 (some outputs violate two). Failing outputs: AboveBelow5 test 2 (palette_exact); AboveBelow6 tests 0, 1, 2 (keep_nonbg); AboveBelow9 test 2 (palette_exact); CleanUp10 tests 1, 2 (nonbg_ge, keep_nonbg); CompleteShape5 tests 1, 2 (palette_subset); CompleteShape6 test 2 (palette_exact); CompleteShape8 test 2 (palette_exact); Copy5 test 2 (palette_exact); Count4 test 2 (palette_subset); ExtractObjects1 tests 1, 2 (palette_exact); FilledNotFilled7 test 2 (palette_exact, palette_subset); HorizontalVertical1 tests 0, 1, 2 (palette_exact); InsideOutside10 test 2 (palette_exact); MoveToBoundary6 test 2 (palette_exact); Order4 tests 1, 2 (nonbg_ge); TopBottom2D9 test 2 (palette_exact). 16 of the 24 failures are on the third test input of their task (8 expected if failures were spread evenly over the three test inputs).

### A12. NVARC candidate pools

Commit run (`runs/commit2`, four evaluation tasks, five outputs): 91 beams, 35 distinct candidates, 31 wrong; one flagged (a wrong near miss violating keep_nonbg), no correct candidate flagged; 3.0/4 for λ in {0, 0.5, 1, 2, ∞}. See `results/nvarc_pool_commit2.json`.

5-hour run (`runs/pools5h`, kernel `chrisvas123/arc-agi-2-nvarc-candidate-pools-5-h` version 1, 2026-09-23): a private copy of the submission notebook with `EVAL_ALL_TASKS = True` and the solver budget cut to five hours, otherwise identical (same model, test-time training, decoding and scoring); not a submission. The solver finished 109 of the 120 public evaluation tasks in sorted task-id order, giving 146 test outputs, 2,188 beams and 1,101 distinct candidates (53 correct, 1,048 wrong). Three further outputs (4a21e3da_1, abc82100_1, b6f77b65_2) are in the pools but have no ground truth in the public ARC-AGI-2 repository copy and are excluded. Pools: `runs/pools5h/nvarc_candidate_pools.tar.gz`; scored by `experiments/nvarc_pool.py` into `results/nvarc_pool_5h.json` with the per-output rows printed in `results/nvarc_pool_5h.log`.

The safe six flag 170 candidates, every one wrong; none of the 53 correct candidates is flagged (ε = 0 on this pool, against 0.8% on the frontier-LLM attempts).

| Candidate class | In pools | Flagged | Flagged % |
|---|---|---|---|
| correct | 53 | 0 | 0.0 |
| near miss | 341 | 59 | 17.3 |
| wrong palette | 205 | 66 | 32.2 |
| wrong content | 376 | 33 | 8.8 |
| wrong shape | 126 | 12 | 9.5 |

Violated families (a candidate can violate several): nonbg_ge 94, palette_exact 67, hist_equal 59, input_in_output 13, keep_nonbg 9, palette_subset 2. 44 of the 146 outputs have at least one flagged candidate.

| λ | Task score /109 | Outputs with a correct top 2 (of 146) | Outputs whose top 2 changed |
|---|---|---|---|
| 0 | 31.667 | 42 | 0 |
| 0.5 | 31.667 | 42 | 2 |
| 1 | 31.667 | 42 | 2 |
| 2 | 31.667 | 42 | 5 |
| ∞ | 32.667 | 43 | 17 |

The shipped rule (λ=1) changes the top 2 on two outputs, neither of which has a correct candidate anywhere in its pool: 9bbf930d_0 (ten flagged near misses, violating hist_equal and nonbg_ge) and c4d067a0_0 (one flagged wrong-content candidate violating keep_nonbg). Its measured effect on the score is therefore zero, as in the commit run. The hard veto changes 17 outputs, gains 31f7f899_0 and loses none, which is the predicted behaviour on an unsaturated pool (kgmon puts a correct candidate in the top 2 for 42 of 146 outputs here, far from the saturation at which vetoes start to cost).

## ARC-AGI-1 (`ATTEMPT_SET=v1 experiments/ablations.py`, `results/ablations_v1.json`)

73 systems, 400 evaluation tasks, 419 test outputs, 56,779 valid attempts (23,900 wrong, 32,879 correct). The safe six flag 23.9% of wrong attempts and 0.40% of correct ones (11.0% of near misses); all sixteen families flag 37.9% and 6.5% (23.7% of near misses). The one failing true output is 19bb5feb (palette_exact), so ε = 1/419 and the hard veto never loses with the safe six. The strongest 2 to 16 systems vote at 98.6 to 99.0 pass@2 and no rule changes them.

### A13. ARC-AGI-1 pooled pass@2 gain over vote, one-vote penalty (points; wins/losses over 200 draws; vote baselines 73.5, 82.8, 92.8, 97.3, 98.2, 98.5)

| Set | 2 | 4 | 8 | 16 | 32 | all 73 |
|---|---|---|---|---|---|---|
| safe6 | +1.03 (164/0) | +2.45 (180/0) | +0.56 (140/0) | +0.10 (59/0) | +0.05 (36/0) | +0.00 (0/0) |
| safe6-palette_exact | +0.82 (155/0) | +1.87 (173/0) | +0.42 (123/0) | +0.08 (51/0) | +0.04 (28/0) | +0.00 (0/0) |
| safe6-palette_subset | +1.01 (163/0) | +2.40 (180/0) | +0.55 (140/0) | +0.10 (59/0) | +0.05 (36/0) | +0.00 (0/0) |
| safe6-hist_equal | +0.95 (161/0) | +2.26 (177/0) | +0.50 (138/0) | +0.09 (54/0) | +0.05 (36/0) | +0.00 (0/0) |
| safe6-nonbg_ge | +0.91 (162/0) | +2.20 (177/0) | +0.47 (127/0) | +0.09 (53/0) | +0.03 (26/0) | +0.00 (0/0) |
| safe6-keep_nonbg | +0.64 (146/0) | +1.58 (171/0) | +0.36 (119/0) | +0.07 (46/0) | +0.04 (31/0) | +0.00 (0/0) |
| safe6-input_in_output | +1.01 (164/0) | +2.43 (180/0) | +0.56 (140/0) | +0.10 (59/0) | +0.05 (36/0) | +0.00 (0/0) |
| safe6+keep_bg | +1.14 (165/0) | +2.68 (183/1) | +0.62 (141/4) | +0.09 (60/13) | +0.03 (34/7) | +0.00 (0/0) |
| safe6+nonbg_equal | +1.08 (164/0) | +2.54 (181/1) | +0.59 (138/4) | +0.08 (59/14) | +0.03 (34/7) | +0.00 (0/0) |
| safe6+nonbg_le | +1.08 (164/0) | +2.56 (181/1) | +0.60 (138/4) | +0.08 (59/14) | +0.03 (34/7) | +0.00 (0/0) |
| safe6+shape | +1.14 (166/0) | +2.70 (183/0) | +0.62 (142/0) | +0.10 (60/0) | +0.05 (36/0) | +0.00 (0/0) |
| safe6+sym_lr | +1.10 (166/0) | +2.59 (181/0) | +0.59 (141/0) | +0.11 (62/0) | +0.05 (39/0) | +0.00 (0/0) |
| safe6+palette_keep | +0.79 (146/14) | +2.31 (173/6) | +0.49 (128/6) | +0.08 (55/6) | +0.04 (35/6) | +0.00 (0/0) |
| all16 | +1.13 (161/13) | +2.99 (180/4) | +0.67 (136/11) | +0.09 (61/20) | +0.03 (35/12) | +0.00 (0/0) |
| shape_only | +0.32 (97/0) | +0.78 (134/1) | +0.12 (61/0) | +0.01 (5/0) | +0.00 (0/0) | +0.00 (0/0) |

### A14. ARC-AGI-1 pooled pass@2 gain over vote, hard veto

| Set | 2 | 4 | 8 | 16 | 32 | all 73 |
|---|---|---|---|---|---|---|
| safe6 | +1.03 (165/0) | +2.50 (181/0) | +0.66 (147/0) | +0.16 (82/0) | +0.12 (84/0) | +0.25 (20/0) |
| safe6-palette_exact | +0.82 (157/0) | +1.89 (173/0) | +0.45 (125/0) | +0.10 (63/0) | +0.10 (74/0) | +0.25 (20/0) |
| safe6-palette_subset | +1.01 (164/0) | +2.45 (181/0) | +0.66 (147/0) | +0.16 (82/0) | +0.12 (84/0) | +0.25 (20/0) |
| safe6-hist_equal | +0.95 (162/0) | +2.31 (178/0) | +0.60 (144/0) | +0.14 (78/0) | +0.12 (84/0) | +0.25 (20/0) |
| safe6-nonbg_ge | +0.92 (163/0) | +2.25 (178/0) | +0.57 (136/0) | +0.13 (73/0) | +0.09 (70/0) | +0.25 (20/0) |
| safe6-keep_nonbg | +0.64 (146/0) | +1.63 (172/0) | +0.44 (128/0) | +0.12 (70/0) | +0.11 (79/0) | +0.25 (20/0) |
| safe6-input_in_output | +1.02 (165/0) | +2.48 (181/0) | +0.66 (147/0) | +0.16 (82/0) | +0.12 (84/0) | +0.25 (20/0) |
| safe6+keep_bg | +1.14 (166/0) | +2.72 (182/3) | +0.67 (136/13) | +0.09 (69/32) | +0.05 (65/31) | +0.25 (20/0) |
| safe6+nonbg_equal | +1.08 (165/0) | +2.58 (181/3) | +0.65 (136/13) | +0.09 (69/33) | +0.05 (65/32) | +0.25 (20/0) |
| safe6+nonbg_le | +1.09 (165/0) | +2.59 (181/3) | +0.62 (129/15) | -0.04 (39/77) | -0.17 (20/120) | -0.25 (0/20) |
| safe6+shape | +1.15 (167/0) | +2.75 (183/0) | +0.68 (145/6) | +0.05 (59/49) | -0.08 (26/91) | +0.00 (0/0) |
| safe6+sym_lr | +1.11 (166/0) | +2.66 (182/0) | +0.73 (148/0) | +0.20 (98/0) | +0.13 (88/0) | +0.25 (20/0) |
| safe6+palette_keep | +0.80 (148/14) | +1.82 (134/44) | -1.04 (19/171) | -2.59 (0/200) | -3.46 (0/200) | -4.00 (0/20) |
| all16 | +1.13 (161/13) | +2.58 (150/33) | -0.77 (33/158) | -2.51 (0/200) | -3.55 (0/200) | -4.25 (0/20) |
| shape_only | +0.33 (98/0) | +0.77 (134/4) | +0.06 (53/28) | -0.13 (5/110) | -0.22 (0/172) | -0.25 (0/20) |

### A15. ARC-AGI-1 gains and losses per output (% of 419 outputs; ε = 1/419)

| Systems | Vote pass@2 | ε x pass@2 | Hard gain | Hard loss | One-vote gain | One-vote loss | Gap to oracle closed (one vote) |
|---|---|---|---|---|---|---|---|
| 2 | 73.5 | 0.18 | 1.00 | 0.00 | 0.99 | 0.00 | 28.2 |
| 4 | 82.8 | 0.20 | 2.40 | 0.00 | 2.35 | 0.00 | 28.0 |
| 8 | 92.8 | 0.22 | 0.63 | 0.00 | 0.53 | 0.00 | 12.0 |
| 16 | 97.3 | 0.23 | 0.15 | 0.00 | 0.10 | 0.00 | 5.7 |
| 32 | 98.2 | 0.23 | 0.11 | 0.00 | 0.04 | 0.00 | 3.4 |
| all 73 | 98.5 | 0.24 | 0.24 | 0.00 | 0.00 | 0.00 | 0.0 |
