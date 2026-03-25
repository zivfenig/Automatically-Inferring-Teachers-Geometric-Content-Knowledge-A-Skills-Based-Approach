# Method II: Multi-Task Learning

This module implements the fine-tuning-based Van Hiele classification pipeline (Method II from the paper). It fine-tunes **Gemma-3-4B-IT** using LoRA (rank 16) and evaluates it on each cross-validation fold.

---

## How It Works

Both variants encode the question-response pair as a Hebrew prompt and pass it through Gemma-3-4B-IT. The last-token hidden state is extracted and fed to a classification head.

**Baseline variant** — trains a single Van Hiele level classification head on top of the frozen (LoRA-adapted) LLM. No skills information is used.

**Skills-aware variant** — adds two components on top of the baseline:

1. **Skills Attention Mechanism**: Each of the 33 skills from the dictionary is represented by a trainable embedding vector (initialized from `multilingual-e5-base` encodings of the skill definitions). Attention weights are computed between the LLM's encoded representation and the skill embeddings. The resulting weighted skill context is concatenated with the LLM representation before classification.

2. **Auxiliary Skills Prediction Head**: A separate multi-label head predicts which skills are demonstrated in the response. It is trained jointly with the classification head using a combined loss:

   ```
   L_total = (1 - λ) · L_level + λ · L_skills
   ```

   where `λ = 0.5` (equal weighting), `L_level` is cross-entropy, and `L_skills` is binary cross-entropy.

---

## Files

| File | Description |
|---|---|
| `baseline_classification.py` | Baseline: classification head only, no skills |
| `skills_variant_classification.py` | Skills-aware: attention + auxiliary skills head |
| `evaluations_and_statistical_tests.ipynb` | Aggregate results and run statistical tests |

---

## Hardware Requirements

- CUDA-capable GPU with **≥24 GB VRAM** (tested on NVIDIA RTX 6000)
- The scripts use `bfloat16` precision and `device_map="auto"` — single GPU is sufficient

---

## Prerequisites

Accept the Gemma license at [huggingface.co/google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) and authenticate:

```bash
huggingface-cli login
```

The model weights are downloaded automatically on first run (~8 GB).

---

## Configuration

Both scripts share the same hyperparameters, set at the top of each file:

| Parameter | Value | Description |
|---|---|---|
| `BASE_MODEL` | `google/gemma-3-4b-it` | Base LLM |
| `FOLD_ID` | `5` | Which fold to train/evaluate (change to 1–5) |
| `LEARNING_RATE` | `2e-4` | AdamW learning rate |
| `WEIGHT_DECAY` | `0.05` | L2 regularization |
| `NUM_EPOCHS` | `30` | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | `4` | Patience on macro F1 |
| `LORA_DROPOUT` | `0.05` | Dropout in LoRA adapters |
| `HEAD_DROPOUT` | `0.25` | Dropout in classification heads |
| `INDICATOR_LOSS_WEIGHT` (λ) | `0.5` | Skills-aware only: auxiliary loss weight |
| `INDICATOR_EMB_DIM` | `512` | Skills-aware only: skill embedding dimension |

---

## Running

**Baseline:**
```bash
cd Multi-Task-Learning
python baseline_classification.py
```

**Skills-Aware:**
```bash
cd Multi-Task-Learning
python skills_variant_classification.py
```

To run all 5 folds, change `FOLD_ID` at the top of the script and rerun. Each fold takes approximately 1–3 hours depending on GPU.

Both scripts support **resuming** from the last checkpoint: if interrupted, simply rerun the same command and training will continue from where it left off.

---

## Output

Results are saved to:
```
results/
├── baseline/
│   └── fold_<N>/
│       ├── checkpoints/           ← Training checkpoints (auto-cleaned, last 3 kept)
│       ├── model/                 ← Final saved model weights + tokenizer
│       │   ├── label_mapping.json
│       │   └── model_config.json
│       └── predictions/
│           ├── val_metrics.json
│           ├── test_metrics.json
│           └── fold_<N>_predictions.csv
│
└── skills_variant/
    └── fold_<N>/                  ← Same structure as baseline
```

The `predictions.csv` contains the ground-truth and predicted Van Hiele levels for each test example.

---

## Evaluation

After running all folds for both variants, open `evaluations_and_statistical_tests.ipynb` to:
- Aggregate F1-macro, F1-weighted, QWK, and MAE across folds
- Run paired t-tests comparing baseline vs. skills-aware
- Reproduce the component ablation analysis (attention-guided vs. skills-supervised vs. full model)
