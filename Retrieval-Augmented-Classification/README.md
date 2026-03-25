# Method I: Retrieval-Augmented Classification

This module implements the RAG-based Van Hiele classification pipeline (Method I from the paper). It uses **Gemini 2.0 Flash** via Google Cloud Vertex AI and the **`multilingual-e5-base`** embedding model for retrieval.

---

## How It Works

Given a new question-response pair, the pipeline:

1. **Embeds** the question and answer using `multilingual-e5-base`, combining them with an 80/20 answer-to-question weighting (reflecting the Van Hiele framework's emphasis on reasoning in responses).
2. **Retrieves** the top-K most similar annotated examples from the training fold using cosine similarity on L2-normalized embeddings.
3. **Builds a prompt** that includes Van Hiele level definitions, (optionally) the skills dictionary, the retrieved examples, and the current question-response pair.
4. **Classifies** by calling Gemini 2.0 Flash and parsing the predicted Van Hiele level (1–5) from the response.

---

## Two Variants

| Variant | Flag | Prompt contents | Description |
|---|---|---|---|
| **A — Baseline** | `variant="A"` | Van Hiele definitions + retrieved Q-A-level examples | No skills information |
| **B — Skills-Aware** | `variant="B"` | Van Hiele definitions + skills dictionary + retrieved Q-A-level-skills examples | Full skills-aware variant |

All other pipeline components (embedding model, retrieval procedure, K, LLM, temperature) are identical between variants.

---

## Directory Structure

```
Retrieval-Augmented-Classification/
├── classify_van_hiele_gemini_RAG.py      ← Main entry point
│
├── rag_mechanism/
│   ├── __init__.py
│   └── retrieve.py                       ← RAGRetriever class
│
├── prompts/
│   ├── prompt_only_definitions.py        ← Variant A prompt builder
│   └── prompt_full_doc_with_indicators.py← Variant B prompt builder
│
├── resources/
│   ├── HE_resources/
│   │   ├── only_definitions.txt          ← Van Hiele level definitions (Hebrew)
│   │   └── Operative_doc_short_version.txt ← Extended doc with skills context (Hebrew)
│   └── EN_resources/
│       ├── only_definitions.txt          ← Van Hiele level definitions (English)
│       └── Operative_doc_short_version.txt ← Extended doc with skills context (English)
│
├── embedding_creation/
│   └── embedd_each_folds.ipynb           ← Recreate embeddings from raw CSVs
│
└── embeddings_folds/
    └── HE_embedded_folds/
        ├── fold_1/
        │   ├── fold_1_train_embeddings.parquet  ← Pre-computed embeddings
        │   ├── fold_1_train_embeddings.csv
        │   └── fold_1_train_RAG_dataset.csv
        ├── fold_2/ ... fold_5/
```

---

## Prerequisites

- GCP project with Vertex AI enabled and access to `gemini-2.0-flash`
- `gcloud` CLI installed and authenticated

```bash
gcloud auth application-default login
```

---

## Configuration

Open `classify_van_hiele_gemini_RAG.py` and update:

```python
PROJECT_ID = "your-gcp-project-id"
LOCATION   = "your-vertex-ai-region"   # e.g., "us-central1"
```

Key experiment constants (already tuned as per the paper):

| Constant | Value | Description |
|---|---|---|
| `K_VALUES` | `[5]` | Number of retrieved examples |
| `ALPHA` | `0.8` | Answer weight in query embedding (80% answer, 20% question) |
| `temperature` | `0.0` | Deterministic generation |

---

## Running

```bash
cd Retrieval-Augmented-Classification
python classify_van_hiele_gemini_RAG.py
```

The `__main__` block defaults to Variant B, fold 1, K=5. To customise:

```python
# Single fold, skills-aware (Variant B)
classify_van_hiele(fold_id=1, variant="B", k=5)

# Single fold, baseline (Variant A)
classify_van_hiele(fold_id=1, variant="A", k=5)

# Full 5-fold cross-validation
for fold_id in [1, 2, 3, 4, 5]:
    classify_van_hiele(fold_id=fold_id, variant="B", k=5)
```

---

## Output

Results are saved to:
```
results/gemini/variant_<A|B>/fold_<N>/gemini_fold<N>_k<K>.csv
```

Each output CSV contains:

| Column | Description |
|---|---|
| `fold` | Fold number |
| `variant` | `A` or `B` |
| `k` | Number of retrieved examples used |
| `test_row_id` | Row index in the test fold |
| `question` | Input question |
| `answer` | Input answer |
| `final_decision` | Ground-truth Van Hiele level |
| `final_indicators` | Ground-truth skill annotations |
| `model_level` | Predicted Van Hiele level |
| `model_explanation` | Model's textual explanation |
| `raw_model_text` | Full raw output from Gemini |

The script saves incrementally every 5 rows and supports **resuming** interrupted runs.

---

## Recreating Embeddings

Pre-computed embeddings are included. If you need to regenerate them (e.g., for a new dataset), run:

```
embedding_creation/embedd_each_folds.ipynb
```

This notebook encodes each training fold using `intfloat/multilingual-e5-base` and saves `.parquet` files to `embeddings_folds/HE_embedded_folds/`.

---

## Evaluation

After running all folds, open `evaluations_and_statistical_tests.ipynb` to:
- Aggregate F1-macro, F1-weighted, QWK, and MAE across folds
- Run paired t-tests comparing Variant A vs. Variant B
- Reproduce the sensitivity analysis (noisy skills experiment)
