# Experiment Log: Prompt Faithfulness Adaptation (PFA → Direct Switch Loss)

## Target: AAAI 2027 (deadline ~August 2026)
## Project: Prompt Faithfulness in 3D Medical SAM

---

## Final Method Configuration (Locked)
```
Method: Direct Switch Loss (Paired Prompt-Contrastive Adaptation)
L = 1.0 * L_seg + 1.0 * L_switch(m=0.2) + 0.5 * L_ground
Epochs: 50
LR: 1e-4, warmup 2 epochs linear + cosine decay to 1e-6
LoRA: 1.9M params (1.89%) on mask decoder + prompt encoder
Base model: SAM-Med3D-turbo (frozen image encoder)
Checkpoint selection: max Dice s.t. Switch >= 0.90
```

---

## Phase 1: Problem Discovery (Gate 1 & 1B)

### Gate 1: Robustness Test (FAILED → pivot)
- **Hypothesis**: SAM-Med3D is fragile to prompt noise
- **Result**: Only 1.3pt Dice drop at shift-10 voxels → model is ROBUST, not fragile
- **Decision**: Pivot from robustness to faithfulness

### Gate 1B: Faithfulness Diagnostic
- **Switch Accuracy (baseline frozen)**: 20.8%
- **Target Margin**: -0.070 (negative = predicts wrong organ)
- **Grounding Rate**: 51.4%
- **Conclusion**: SAM-Med3D largely IGNORES prompt location. Problem is controllability, not robustness.

---

## Phase 2: Initial Method Attempts (FAILED)

### Old PFA Losses (L_ground + L_contrast + L_stability)
| Method | Ep5 Switch | Ep5 Dice | Notes |
|--------|-----------|----------|-------|
| seg_only | 0.542 | 0.550 | Simple LoRA baseline |
| pfa_full | 0.500 | 0.540 | PFA worse than seg_only |
| McNemar p-value | 0.68 | — | Not significant |

**Diagnosis**: L_contrast saturates to ~0.006 by epoch 1 (margin too easy). No persistent gradient signal.

### Ablation: ground_only
| Epoch | Switch | Dice |
|-------|--------|------|
| 1 | 0.700 | 0.448 |
| 2 | 0.460 | 0.540 |

**Pattern**: High Switch early, decays as Dice improves. Same as pfa_full.

### Curriculum Training (FAILED)
Config: Phase 1 (ep1-2) 0.2*seg + strong PFA, Phase 2 (ep3-5) 0.7*seg + stronger PFA
| Epoch | Phase | Switch | Dice |
|-------|-------|--------|------|
| 1 | 1 | 0.656 | 0.357 |
| 2 | 1 | 0.532 | 0.422 |
| 3 | 2 | 0.427 | 0.485 |
| 4 | 2 | 0.516 | 0.516 |

**Key finding**: Switch decays EVEN within Phase 1 (low seg weight). Problem is NOT seg loss overwriting PFA — it's that L_contrast provides no persistent gradient.

---

## Phase 3: Degeneracy Check (ep1 validation)

| Checkpoint | Dice | Switch (all) | Switch (filtered) | Vol Ratio | Tiny/Empty |
|---|---|---|---|---|---|
| pfa_full ep1 | 0.451 | 0.604 | 0.760 | 1.03 | 0/0 |
| pfa_full ep5 | 0.529 | 0.500 | 0.800 | 1.17 | 0/0 |
| seg_only ep1 | 0.476 | 0.447 | 0.522 | 1.31 | 0/0 |
| seg_only ep5 | 0.570 | 0.516 | 0.862 | 1.04 | 0/0 |

**Conclusion**: No degeneracy. PFA ep1 has real faithfulness signal with acceptable Dice. The problem is purely training dynamics.

---

## Phase 4: Direct Switch Loss (BREAKTHROUGH)

### Design Rationale
- L_contrast fails because margin=0.1 is too easy → saturates to ~0.006
- Direct Switch Loss uses paired prompts: prompt_A→pred_A, prompt_B→pred_B
- Loss enforces: Dice(pred_A, GT_A) > Dice(pred_A, GT_B) + margin (and vice versa)
- Margin=0.2 keeps gradient signal alive (L_switch stays ~0.02-0.09 throughout training)

### 5-Epoch Pilot (old config: 0.7*seg + 2.0*switch)
| Epoch | Switch | Dice | L_switch |
|-------|--------|------|----------|
| 1 | 0.979 | 0.374 | 0.091 |
| 2 | 0.979 | 0.430 | 0.052 |
| 3 | 0.971 | 0.470 | 0.048 |
| 4 | 0.967 | 0.509 | 0.045 |
| 5 | 0.970 | 0.444 | 0.046 |

**Key**: Switch stays ~0.97 with NO decay. First method to maintain faithfulness across epochs.

---

## Phase 5: Formal 50-Epoch Experiments (CURRENT)

### AMOS - Main Results (Best checkpoint: max Dice s.t. Switch >= 0.90)

| Method | Seed | Best Dice | Best Switch | Best Epoch |
|--------|------|-----------|-------------|------------|
| switch_loss | 42 | 0.677 | 1.000 | 47 |
| switch_loss | 123 | 0.683 | 1.000 | 47 |
| switch_loss | 7 | (running) | — | — |
| seg_only | 42 | 0.723 | 0.462 | 47 |
| seg_only | 123 | 0.726 | 0.462 | 47 |
| seg_only | 7 | (running) | — | — |
| frozen baseline | — | ~0.45 | 0.208 | — |

### Epoch Curves (seed=42)
| Epoch | switch_loss Switch | switch_loss Dice | seg_only Switch | seg_only Dice |
|-------|-------------------|-----------------|-----------------|---------------|
| 1 | 0.958 | 0.346 | 0.604 | 0.426 |
| 5 | 0.980 | 0.477 | — | — |
| 10 | 0.971 | 0.544 | 0.573 | 0.611 |
| 15 | 1.000 | 0.560 | — | — |
| 20 | 0.990 | 0.586 | 0.440 | 0.652 |
| 30 | 1.000 | 0.650 | 0.553 | 0.699 |
| 40 | 0.979 | 0.666 | 0.547 | 0.715 |
| 50 | 0.978 | 0.641 | 0.549 | 0.698 |

### Summary Statistics (2 seeds completed)
- **switch_loss**: Dice = 0.680 ± 0.003, Switch = 1.000 ± 0.000
- **seg_only**: Dice = 0.725 ± 0.002, Switch = 0.462 ± 0.000
- **Dice gap**: ~4.5pt (acceptable per Codex review: ≤5pt is publishable)
- **Switch gap**: +53.8pt (massive improvement)

---

## Phase 6: Ablations & Pareto Sweep (20 epochs, seed=42)

### Completed Ablations

| Config | Best Dice | Best Switch | Best Epoch | Final Dice (ep20) | Final Switch (ep20) |
|--------|-----------|-------------|------------|-------------------|---------------------|
| margin=0.1 (m=0.1, λ_sw=1.0) | 0.602 | 0.989 | 18 | 0.561 | 0.990 |
| margin=0.3 (m=0.3, λ_sw=1.0) | 0.594 | 0.980 | 18 | 0.555 | 0.990 |
| no_ground (m=0.2, λ_sw=1.0, λ_gnd=0) | — | 0.980 | — | 0.571 | 0.980 |
| λ_switch=0.25 (m=0.2, λ_sw=0.25) | 0.608 | 0.989 | 12 | 0.567 | 0.990 |
| λ_switch=0.5 (m=0.2, λ_sw=0.5) | 0.603 | 0.978 | 12 | 0.562 | 0.990 |
| λ_switch=2.0 (m=0.2, λ_sw=2.0) | — | 0.990 | — | 0.545 | 0.990 |
| paired_seg_only (no switch loss) | (running) | — | — | — | — |

### Epoch Curves (completed)

**margin=0.1:**
| Ep | Switch | Dice |
|----|--------|------|
| 1 | 0.958 | 0.357 |
| 10 | 0.981 | 0.545 |
| 20 | 0.990 | 0.561 |

**margin=0.3:**
| Ep | Switch | Dice |
|----|--------|------|
| 1 | 0.958 | 0.357 |
| 10 | 0.971 | 0.539 |
| 20 | 0.990 | 0.555 |

**λ_switch=0.25:**
| Ep | Switch | Dice |
|----|--------|------|
| 1 | 0.958 | 0.357 |
| 10 | 0.971 | 0.548 |
| 20 | 0.990 | 0.567 |

**λ_switch=0.5:**
| Ep | Switch | Dice |
|----|--------|------|
| 1 | 0.958 | 0.357 |
| 10 | 0.971 | 0.546 |
| 20 | 0.990 | 0.562 |

### Preliminary Pareto Observations
- ALL configs maintain Switch ≥ 0.978 regardless of margin or λ_switch
- Method is extremely robust to hyperparameters
- λ_switch=0.25 gives highest best Dice (0.608) — lower switch weight helps Dice without hurting Switch
- margin=0.1 vs 0.2 vs 0.3: minimal difference (best Dice 0.602/main/0.594)
- Direct Switch Loss works even at very low weight (0.25) — the paired contrastive signal is inherently strong

---

## Phase 7: Mechanism Ablation (Critical Discovery)

### 2×2 Factor Isolation Matrix (20 epochs, seed=42)

| Method | Paired? | Ground? | Switch Loss? | Switch Acc | Dice |
|--------|---------|---------|--------------|-----------|------|
| seg_only | ❌ | ❌ | ❌ | 0.55 | 0.62 |
| single_ground | ❌ | ✅ | ❌ | 0.43 | 0.624 |
| paired_no_ground | ✅ | ❌ | ❌ | 0.94 | 0.597 |
| paired_seg_only | ✅ | ✅ | ❌ | 0.99 | 0.573 |
| switch_loss (full) | ✅ | ✅ | ✅ | 0.99 | 0.595 |

### Key Finding
**Paired prompt training is the core mechanism (+40pt Switch).**
- Single prompt ± grounding: Switch 0.43-0.55 (insufficient, grounding alone even HURTS)
- Paired prompt ± grounding ± switch: Switch 0.94-0.99 (all work)
- Grounding adds ~5pt on top of paired training (0.94 → 0.99)
- Switch margin loss adds ~0pt on top of paired + grounding (0.99 → 0.99)
- single_ground DECREASES Switch (0.55 → 0.43) — grounding without paired training is counterproductive

### Implication for Paper Story
The contribution is NOT "a specific loss function" but rather:
1. Diagnosing prompt unfaithfulness (Switch Accuracy metric)
2. Identifying the root cause (single-prompt training never constrains counterfactual behavior)
3. Fixing it through paired prompt-contrastive training paradigm

---

## Computational Overhead Analysis

| Metric | seg_only | Paired Training (ours) |
|--------|----------|----------------------|
| Total params | 100.5M | 100.5M (same) |
| Trainable params | 1.9M (1.89%) | 1.9M (1.89%) |
| Training time/epoch | 1117s | 1177s (+5.3%) |
| Total training (50ep) | 6.2h | 6.5h |
| Inference time | identical | identical |
| Checkpoint size | 7.6MB | 7.6MB |

**Overhead is minimal: +5.3% training time, zero inference overhead.**

---

## Paper Figure and Table Plan

Local path: `E:\??\PEA-MedSeg\figures\`
Reference style pages: `E:\??\PEA-MedSeg\figures\reference_style\`
Tables: `E:\??\PEA-MedSeg\paper_assets\tables.md`

| Item | File | Content | Status |
|------|------|---------|--------|
| Fig. 1 | fig1_problem_overview.svg | Method overview / problem illustration in the compact composite style of the reference papers. Shows expected prompt-controlled behavior, SAM-Med3D unfaithfulness, paired training fix, and Switch Accuracy definition. | Draft SVG |
| Fig. 2 | fig2_training_framework.svg | Training architecture for Paired Prompt-Contrastive Training: paired input, frozen image encoder, two prompt-conditioned LoRA decoder paths, and L_seg/L_switch/L_ground. | Draft SVG |
| Fig. 3 | fig3_dice_switch_curves.svg | Dice-Switch training curves using logged epoch points for switch_loss vs seg_only. | Draft SVG from actual log values |
| Fig. 4 | fig4_qualitative_prompt_switch.svg | Qualitative prompt A vs B comparison template. Final version must replace schematic CT panels with fixed-prompt-bank real CT crops. | Draft SVG template |
| Fig. 5 | fig5_ablation_pareto.svg | Ablation/Pareto figure using completed 20-epoch sweep and mechanism ablation values. Approximate mechanism values are marked with ~. | Draft SVG from actual/approx log values |
| Table 1 | paper_assets/tables.md | Main AMOS/BTCV results. | Draft |
| Table 2 | paper_assets/tables.md | Ablation table. | Draft |
| Table 3 | paper_assets/tables.md | Per-organ analysis template. Values are intentionally TBD because per-organ analysis is still running. | Template only |

**Style target:** follow the two provided reference papers' figure language: compact multi-panel composition, pale tinted section backgrounds, thin black module borders, short arrows, small medical-image thumbnails, subfigure labels (a)(b)(c), and high information density. Avoid sparse poster-like diagrams. Use real experiment values for plots/tables; do not fabricate per-organ results.

---

## Experiment Checklist (Updated)

### Tier 1: Main Claim (MANDATORY)
- [x] Frozen SAM-Med3D baseline eval on AMOS
- [x] AMOS seg_only 50ep seed=42
- [x] AMOS seg_only 50ep seed=123
- [x] AMOS seg_only 50ep seed=7
- [x] AMOS switch_loss 50ep seed=42
- [x] AMOS switch_loss 50ep seed=123
- [x] AMOS switch_loss 50ep seed=7

### Tier 2: Ablations (20 epochs, 1 seed)
- [x] margin=0.1 → Best Dice=0.602, Switch=0.989
- [x] margin=0.3 → Best Dice=0.594, Switch=0.980
- [x] no L_ground → Dice=0.571, Switch=0.980
- [x] λ_switch=2.0 → Dice=0.545, Switch=0.990
- [ ] paired_seg_only (mechanism ablation, RUNNING)
- [ ] switch_loss 20ep anchor (RUNNING)
- [ ] seg_only 20ep anchor (RUNNING)

### Tier 3: Pareto Sweep (20 epochs, 1 seed)
- [x] lambda_switch=0.25 → Best Dice=0.608, Switch=0.989
- [x] lambda_switch=0.5 → Best Dice=0.603, Switch=0.978
- [x] lambda_switch=2.0 → Dice=0.545, Switch=0.990

### Tier 4: External Validation
- [x] BTCV switch_loss 50ep → Best Dice=0.704, Switch=1.000 (ep30)
- [x] BTCV seg_only 50ep → Best Dice=0.745, Switch=0.312

### Tier 5: Model Transfer
- [ ] SegVol baseline Switch test (deferred - version incompatibility)

### Tier 6: Cross-Domain
- [x] SAM2 on COCO → Switch=0.954 (already faithful, no problem to fix)

### P0 Supplementary (Codex final review requirements)
- [ ] Fixed prompt bank re-evaluation (RUNNING)
- [ ] Statistical tests (paired t-test/McNemar) (RUNNING with above)
- [ ] Per-organ Dice/Switch analysis (RUNNING with above)
- [ ] paired_seg_only mechanism ablation (RUNNING)

### P1 Supplementary
- [x] Qualitative visualization draft replaced with paper-style prompt A/B switch figures
- [x] Computational overhead analysis (params, time)

### P2 Supplementary
- [ ] Pareto sweep λ=0.25 at 50ep (if Dice gap needs closing)

---

## Key Decisions (Codex Adversarial Review, 3 rounds)

1. **Pivot from PFA losses to Direct Switch Loss** — L_contrast saturates too fast (0.006 by ep1), provides no persistent gradient. Direct Switch Loss stays at 0.02-0.09 throughout.
2. **Constant weights, no scheduling** — Simpler, more defensible for reviewers. Scheduling adds complexity without clear benefit.
3. **50 epochs** — 5ep pilot showed Dice not converged. 50ep gives full convergence.
4. **L_seg=1.0, L_switch=1.0** — Equal weighting. Pilot used 0.7:2.0 (too switch-heavy, capped Dice).
5. **Checkpoint selection: max Dice s.t. Switch≥0.90** — Defensible constrained optimization.
6. **Pareto sweep as contingency** — If Dice gap >5pt, show frontier with different λ_switch values.
7. **Drop L_stability** — May oppose switching behavior, not central to method.

---

## Server Info
- Server: ssh -p 14256 root@connect.nmb1.seetacloud.com
- Code: /root/autodl-tmp/PEA-MedSeg/
- Data: /root/autodl-tmp/data/ (amos22_cached, btcv_cached)
- GPU0 + GPU1: 2×RTX 4090
- Results: /root/autodl-tmp/PEA-MedSeg/results/formal/
