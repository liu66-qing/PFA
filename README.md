# PFA: Prompt Faithfulness Adaptation for SAM-Med3D

Diagnosing and fixing prompt unfaithfulness in 3D medical Segment Anything models via paired prompt-contrastive training.

## Problem

SAM-Med3D achieves reasonable segmentation quality but **ignores point prompts** — it predicts the same dominant organ regardless of where you click. We measure this with **Switch Accuracy**: when you switch the prompt from organ A to organ B, does the prediction actually change?

- SAM-Med3D baseline Switch Accuracy: **20.8%** (nearly random)
- After PFA training: **98.5%**

## Method

- **Base model**: SAM-Med3D-turbo (frozen image encoder)
- **Adaptation**: LoRA on mask decoder + prompt encoder (1.9M params, 1.89% of total)
- **Training signal**: Paired prompt-contrastive loss
  - Two prompts per image targeting different organs
  - Direct Switch Loss forces different predictions for different prompts
  - Grounding loss ensures prompt point falls inside predicted mask

## Key Results

| Method | Dice | Switch Acc | Training Overhead |
|--------|------|-----------|-------------------|
| seg-only (baseline) | 0.720 | 0.208 | - |
| Unpaired multi-prompt | 0.614 | 0.776 | - |
| **PFA (ours)** | **0.676** | **0.985** | +5.3% |

## Repository Structure

```
src/                    # Core source code
  pea_dataset.py        # Multi-organ prompt pair sampling
  pfa_loss.py           # Segmentation + grounding loss
  train_pfa.py          # Training and inference functions
  lora.py               # LoRA adaptation for SAM-Med3D
  eval_faithfulness_v2.py  # Switch Accuracy evaluation
  sam_med3d_gate1_wrapper.py  # Model loading wrapper

run_formal_experiment.py    # Main training script (PFA + baselines)
run_unpaired_baseline.py    # Unpaired multi-prompt control experiment
run_hard_prompt_eval.py     # Hard prompt evaluation benchmark
run_per_organ_eval.py       # Per-organ analysis + runtime profiling
run_ablation.py             # Mechanism ablation experiments

paper/                  # LaTeX source (AAAI format, 8 pages)
results/                # Experiment result JSONs
checkpoints/            # LoRA weights (see below)
```

## Checkpoints

LoRA weights (7.3MB each) are available at: [Hugging Face](https://huggingface.co/jiujiu66/PFA)

| Checkpoint | Description |
|-----------|-------------|
| `switch_loss_seed42/best.pth` | PFA (ours), best model |
| `switch_loss_seed{123,7}/best.pth` | PFA, other seeds |
| `seg_only_seed42/epoch050.pth` | Baseline (seg-only) |
| `no_ground_seed42/best.pth` | Ablation: no grounding loss |
| `single_ground_seed42/best.pth` | Ablation: grounding only |
| `paired_no_ground_seed42/best.pth` | Ablation: paired without grounding |
| `paired_seg_only_seed42/best.pth` | Ablation: paired seg-only |

**Base model required**: [SAM-Med3D-turbo](https://github.com/uni-medical/SAM-Med3D) checkpoint (`sam_med3d_turbo.pth`)

## Datasets

| Dataset | Usage | Link |
|---------|-------|------|
| AMOS22 | Primary training and evaluation (13 abdominal organs) | [AMOS Challenge](https://amos22.grand-challenge.org/) / [Zenodo](https://zenodo.org/records/7155725) |
| BTCV | Cross-dataset validation | [Synapse Multi-organ](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) |

**Data preparation**: See `src/cache_data.py` for preprocessing (resampling to 1.5mm isotropic, intensity normalization, caching as .npy).

## Requirements

```
torch >= 2.0
numpy
scipy
SimpleITK
monai (optional, for data loading)
```

## Usage

### Training PFA

```python
# Main experiment (paired prompt-contrastive training)
python run_formal_experiment.py

# Unpaired baseline (control experiment)
python run_unpaired_baseline.py
```

### Evaluation

```python
# Per-organ Dice + Switch Accuracy
python run_per_organ_eval.py

# Hard prompt benchmark (boundary, noisy, adjacent-organ)
python run_hard_prompt_eval.py
```

## Citation

```bibtex
@article{liu2026pfa,
  title={Diagnosing and Fixing Prompt Unfaithfulness in {SAM-Med3D} via Paired Prompt-Contrastive Training},
  author={Liu, Junqing},
  year={2026}
}
```

## License

MIT
