# Paper Improvement Log

## Paper
**Title**: Diagnosing and Fixing Prompt Unfaithfulness in SAM-Med3D via Paired Prompt-Contrastive Training
**Author**: Junqing Liu
**Venue Target**: AAAI 2027
**Format**: 8 pages main body, two-column, AAAI style
**Date**: 2026-05-25

## Score Progression

| Round | Score | Verdict | Key Changes |
|-------|-------|---------|-------------|
| Round 0 (original) | ~5/10 | No | Overclaiming title, missing prompt bank details, no per-organ analysis, overfull hbox |
| Round 1 | 6/10 | Almost | Fixed formatting, added figures, expanded content to 8 pages, updated refs to 2024-2025 |
| Round 2 (final) | 7/10 | Almost | Narrowed title/claims, added prompt bank construction, training budget clarification, per-organ analysis, GT-derived limitation |

## Round 1 Review & Fixes

<details>
<summary>GPT-5.5 Review (Round 1) — Score: 6/10</summary>

**Summary**: The paper identifies an important and under-evaluated failure mode in promptable 3D medical segmentation. The proposed Switch Accuracy metric is intuitive, and the paired prompt training intervention is simple, practical, and apparently very effective. However, the current evidence is too narrow for the breadth of the claims.

**CRITICAL Issues:**
1. "Root cause is single-prompt training" claim is under-proven — need stronger causal evidence
2. Switch Accuracy may be too easy or partially degenerate — need harder switch tests
3. Generalization claim too broad for one external dataset
4. Single-architecture limitation is serious
5. Dice drop needs deeper analysis

**MAJOR Issues:**
1. Need comparisons to obvious alternatives (prompt dropout, multi-prompt without pairing)
2. Clarify train/test prompt bank construction
3. Switch Accuracy should be reported with confidence intervals and per-seed details
4. COCO SAM2 comparison may not support stated conclusion
5. Mechanism ablation table underspecified (training budget)
6. Novelty framing should be more precise

**MINOR Issues:**
1. Title may overclaim
2. Define Switch Accuracy early and visually
3. Be careful with percentages vs points
4. Clarify Dice notation in switch loss
5. Grounding loss notation underspecified
6. Discuss clinical relevance carefully
7. Use limitations more strategically
8. 8-page utilization

</details>

### Fixes Implemented (Round 1 → Round 2)
1. **Title narrowed**: "Restores Prompt Faithfulness in 3D Medical Segment Anything Models" → "Diagnosing and Fixing Prompt Unfaithfulness in SAM-Med3D via Paired Prompt-Contrastive Training"
2. **Prompt bank construction detailed**: Added erosion by 3 voxels, fixed seed, per-seed Switch values (0.985, 1.000, 0.994), coverage of adjacent organs
3. **Training budget clarified**: Same optimizer steps, same images per epoch; seg-only at 100ep still only 0.55 Switch
4. **Cross-domain conclusion narrowed**: Removed "specific to 3D medical adaptation" → "not inherent to promptable segmentation paradigm"
5. **Per-organ Dice-Switch analysis added**: Size-dependent pattern (large >0.80, medium 0.65-0.75, small 0.40-0.55), hardest pairs >0.83
6. **Evaluation practice claim softened**: "should become standard" → "should complement standard Dice evaluation"
7. **New limitation added**: GT-derived prompts, near-saturated metric caveat
8. **Contribution 4 reworded**: Avoid overclaiming about 3D-medical specificity

## Round 2 Review & Fixes

<details>
<summary>GPT-5.5 Review (Round 2) — Score: 7/10</summary>

**Summary**: The revision moves the paper from a borderline/overclaimed 6/10 to a solid focused contribution around 7.0/10. The title is now appropriately narrowed. Prompt bank construction is much more reproducible. The training-budget clarification removes a major ambiguity. Per-organ analysis adds useful diagnostic depth.

**Remaining Major Issues:**
1. Single-architecture evidence remains limiting (acceptable for 7, limits confidence above 7.5)
2. Prompt evaluation is still relatively easy (GT-derived, interior points)
3. Paired training has a supervision advantage (honest but not perfectly controlled)
4. No stronger external baselines

**To push to 7.5+:**
- Add harder prompt benchmark (boundary clicks, noisy clicks, adjacent-organ-only)
- Add one nontrivial baseline (prompt dropout, unpaired multi-prompt)
- Evaluate on another architecture
- Provide full per-organ tables with confidence intervals
- Include compute/runtime overhead analysis

**Verdict**: Accept-leaning if venue values focused empirical diagnosis. Not yet strong accept due to narrow scope and near-saturated evaluation.

</details>

### Assessment
Round 2 issues are primarily scope/experiment limitations that require additional compute, not writing fixes. The paper is at its writing-quality ceiling given current experimental evidence.

## Format Check

| Metric | Value |
|--------|-------|
| Pages (main body) | 8 |
| LaTeX errors | 0 |
| Overfull hbox | 0 |
| Undefined references | 0 |
| Undefined citations | 0 |

## PDFs
- `main_round0_original.pdf` — First compiled draft (5 pages, formatting issues)
- `main_round1_draft.pdf` — After content expansion (7 pages, pre-review)
- `main_round2.pdf` — Final version after 2 rounds of review fixes (8 pages)
- `main.pdf` — = main_round2.pdf (current)

## Recommendations for Further Improvement (Experiment-Level)

To reach 7.5+/10, the following experiments would be needed:
1. **Harder prompt evaluation**: Boundary-near clicks, noisy clicks (±5 voxels), adjacent-organ-only pairs
2. **Additional baseline**: Unpaired multi-prompt training (same exposure, no simultaneous pairing)
3. **Second architecture**: SAM-Med3D full checkpoint or MedSAM-style 3D adaptation
4. **Full per-organ table**: All 13 organs × {Dice, Switch} × {seg-only, ours} with CI
5. **Runtime table**: GPU memory, wall-clock per step, FLOPs comparison
