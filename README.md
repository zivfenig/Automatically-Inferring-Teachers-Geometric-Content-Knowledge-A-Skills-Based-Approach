# Automatically Inferring Teachers' Geometric Content Knowledge: A Skills-Based Approach

**AIED 2026** · Ziv Fenigstein, Kobi Gal, Avi Segal, Osama Swidan, Inbal Israel, Hassan Ayoob  
Ben-Gurion University of the Negev · University of Edinburgh

> This repository contains the code, data, and resources accompanying the paper *"Automatically Inferring Teachers' Geometric Content Knowledge: A Skills Based Approach"*, accepted at the **AIED 2026** conference.

---

## Overview

This work presents an automated approach for classifying teachers' Van Hiele levels of geometric reasoning from open-ended, free-text responses in Hebrew. The central hypothesis is that explicitly integrating a structured **skills dictionary** — which decomposes each Van Hiele level into 33 fine-grained reasoning skills — significantly improves classification over baselines that rely on Van Hiele level definitions alone.

Two classification methods are implemented and compared:

| Method | Variant | Description |
|---|---|---|
| **Method I: RAG** | Baseline (A) | Retrieval-Augmented Generation with Van Hiele definitions only |
| **Method I: RAG** | Skills-Aware (B) | RAG augmented with the full skills dictionary and annotated skill labels |
| **Method II: MTL** | Baseline | Fine-tuned Gemma-3-4B with a classification head; no skills information |
| **Method II: MTL** | Skills-Aware | Fine-tuned Gemma-3-4B with skills attention mechanism + auxiliary skills prediction task |

Both skills-aware variants significantly outperform their respective baselines across all evaluation metrics (F1-macro, F1-weighted, QWK, MAE) in 5-fold cross-validation.

---

## Repository Structure

```
AIED_REPO_FOLDER/
│
├── README.md                          ← You are here
├── requirements.txt                   ← Python dependencies
│
├── Data-and-preprocess/               ← Dataset, skills dictionary, fold creation
│   ├── README.md
│   ├── create_folds.ipynb             ← Notebook to reproduce the 5-fold split
│   ├── HE_Van_Hiele_Dataset/          ← Hebrew dataset (226 Q-A pairs)
│   │   ├── combined.csv               ← Full annotated dataset
│   │   └── folds/                     ← Pre-split fold CSVs (fold_N_train/test.csv)
│   ├── EN_Van_Hiele_Dataset/
│   │   └── combined_english.csv       ← English translation of the dataset
│   ├── HE_Skills_dictionary/
│   │   └── indicators_dictionary.py   ← 33 reasoning skills in Hebrew
│   └── EN_Skills_dictionary/
│       └── indicators_dictionary_english.py  ← 33 reasoning skills in English
│
├── Retrieval-Augmented-Classification/  ← Method I (RAG)
│   ├── README.md
│   ├── classify_van_hiele_gemini_RAG.py  ← Main entry point
│   ├── rag_mechanism/
│   │   └── retrieve.py                ← RAGRetriever class
│   ├── prompts/
│   │   ├── prompt_only_definitions.py ← Variant A prompt builder (baseline)
│   │   └── prompt_full_doc_with_indicators.py  ← Variant B prompt builder (skills-aware)
│   ├── resources/
│   │   ├── HE_resources/              ← Hebrew Van Hiele definitions & operative doc
│   │   └── EN_resources/              ← English versions
│   ├── embedding_creation/
│   │   └── embedd_each_folds.ipynb    ← Notebook to recreate embeddings per fold
│   └── embeddings_folds/
│       └── HE_embedded_folds/         ← Pre-computed embeddings for folds 1–5
│
└── Multi-Task-Learning/               ← Method II (MTL)
    ├── README.md
    ├── skills_variant_classification.py  ← Skills-aware MTL training + evaluation
    ├── baseline_classification.py        ← Baseline training + evaluation
    └── evaluations_and_statistical_tests.ipynb
```

---

## Dataset

The dataset consists of **226 question-response pairs** from **31 pre-service mathematics teachers** at three Israeli teacher-training institutions. Each pair is annotated with:
- A **Van Hiele level** (1–5), assigned by two independent expert annotators (inter-rater Cohen's κ = 0.84)
- The **reasoning skills** demonstrated in the response, selected from the 33-skill dictionary

The dataset is written in **Hebrew**. An English translation is provided in `EN_Van_Hiele_Dataset/`.

**Label distribution:**

| Level | Name | % of Dataset |
|---|---|---|
| 1 | Visualization | 19% |
| 2 | Analysis | 25% |
| 3 | Informal Deduction | 35% |
| 4 | Deduction | 14% |
| 5 | Rigor | 7% |

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (required for Method II; ≥24 GB VRAM recommended for Gemma-3-4B)
- A Google Cloud Platform project with Vertex AI enabled (required for Method I)

### Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

pip install -r requirements.txt
```

---

## Running Method I: Retrieval-Augmented Classification

Method I uses **Gemini 2.0 Flash** via Google Cloud Vertex AI. Before running:

**Step 1 — Authenticate with GCP:**
```bash
gcloud auth application-default login
```

**Step 2 — Set your project credentials** in `Retrieval-Augmented-Classification/classify_van_hiele_gemini_RAG.py`:
```python
PROJECT_ID = "your-gcp-project-id"
LOCATION   = "your-vertex-ai-region"   # e.g., "us-central1"
```

**Step 3 — Run classification:**

Pre-computed `multilingual-e5-base` embeddings for all 5 folds are already included. Run from the repo root:

```bash
cd Retrieval-Augmented-Classification
python classify_van_hiele_gemini_RAG.py
```

By default the script runs **Variant B (skills-aware)** on fold 1 with K=5. Edit the `__main__` block to run other configurations:

```python
# Variant B (skills-aware) – single fold
classify_van_hiele(fold_id=1, variant="B", k=5)

# Variant A (baseline) – single fold
classify_van_hiele(fold_id=1, variant="A", k=5)

# Full 5-fold cross-validation, skills-aware
for fold_id in [1, 2, 3, 4, 5]:
    classify_van_hiele(fold_id=fold_id, variant="B", k=5)
```

Results are saved to `Retrieval-Augmented-Classification/results/gemini/variant_<A|B>/fold_<N>/`.

The script supports **resuming**: if interrupted, re-running will skip already-processed rows.

See `Retrieval-Augmented-Classification/README.md` for full details.

---

## Running Method II: Multi-Task Learning

Method II fine-tunes **Gemma-3-4B-IT** locally. A CUDA GPU with at least 24 GB VRAM is required.

**Step 1 — Accept the Gemma license** at [huggingface.co/google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it), then authenticate:
```bash
huggingface-cli login
```

**Step 2 — Run the desired variant:**

Skills-aware variant:
```bash
cd Multi-Task-Learning
python skills_variant_classification.py
```

Baseline variant:
```bash
cd Multi-Task-Learning
python baseline_classification.py
```

Both scripts default to `FOLD_ID = 5`. Edit `FOLD_ID` at the top of the file to run a specific fold (1–5). Run each script once per fold to reproduce full 5-fold cross-validation.

Results (metrics JSON + predictions CSV) are saved to `Multi-Task-Learning/results/<baseline|skills_variant>/fold_<N>/`.

See `Multi-Task-Learning/README.md` for full details.

---

## Evaluation & Statistical Tests

After collecting results across all folds, use the evaluation notebooks to aggregate metrics and reproduce the paired t-tests from the paper:

- `Retrieval-Augmented-Classification/evaluations_and_statistical_tests.ipynb`
- `Multi-Task-Learning/evaluations_and_statistical_tests.ipynb`

---

## Results Summary

Average results across 5-fold cross-validation (from the paper):

| Method | Variant | F1-macro | F1-weighted | QWK | MAE ↓ |
|---|---|---|---|---|---|
| RAG | Baseline (A) | 0.624 | 0.678 | 0.625 | 0.470 |
| RAG | Skills-Aware (B) | **0.695** | **0.736** | **0.721** | **0.376** |
| MTL | Baseline | 0.646 | 0.657 | 0.586 | 0.523 |
| MTL | Skills-Aware | **0.725** | **0.725** | **0.717** | **0.403** |

For MAE, lower is better. For all other metrics, higher is better.

---

## Citation

If you use this code or data in your research, please cite:

```bibtex

```

---


