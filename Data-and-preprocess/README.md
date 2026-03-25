# Data and Preprocessing

This folder contains the annotated dataset, the skills dictionary, and the fold-creation notebook used in the paper.

> **Language note:** This research was conducted in Israel with Israeli pre-service mathematics teachers. All data collection, teacher responses, and expert annotations were carried out in **Hebrew**. The Hebrew versions (`HE_*`) are the primary data used in all experiments and results reported in the paper. The English versions (`EN_*`) are provided as translations for the broader research community.

---

## Contents

```
Data-and-preprocess/
├── create_folds.ipynb             ← Reproduces the 5-fold stratified split
│
├── HE_Van_Hiele_Dataset/          ← PRIMARY dataset used in the paper (Hebrew)
│   ├── combined.csv               ← Full annotated dataset (226 rows)
│   └── folds/
│       ├── fold_1_train.csv
│       ├── fold_1_test.csv
│       ├── ...
│       ├── fold_5_train.csv
│       └── fold_5_test.csv
│
├── EN_Van_Hiele_Dataset/
│   └── combined_english.csv       ← English translation of the full dataset (for reference)
│
├── HE_Skills_dictionary/
│   └── indicators_dictionary.py   ← 33 Van Hiele reasoning skills in Hebrew (used in the paper)
│
└── EN_Skills_dictionary/
    └── indicators_dictionary_english.py  ← English translation of the skills dictionary (for reference)
```

---

## Dataset: `combined.csv` / `combined_english.csv`

Each row is one question-response pair annotated by expert mathematics educators.

| Column | Description |
|---|---|
| `question` | Open-ended geometry question posed to the teacher (Hebrew / English) |
| `answer` | Teacher's free-text response |
| `final_decision` | Expert-assigned Van Hiele level (integer 1–5) |
| `final_indicators` | Comma-separated skill codes demonstrated in the response (e.g., `"2.01, 2.06"`) |

**226 total pairs** from **31 pre-service teachers** across three Israeli institutions. The annotation process used a double-blind protocol with two independent experts (Cohen's κ = 0.84 for Van Hiele levels).

---

## Cross-Validation Folds

The 5-fold split was created with stratified sampling (preserving Van Hiele level distribution) and a fixed random seed (42) for reproducibility. The folds are pre-generated and stored in `HE_Van_Hiele_Dataset/folds/`. Each fold CSV contains the same columns as `combined.csv`.

To regenerate the folds from scratch, run `create_folds.ipynb`.

---

## Skills Dictionary

The skills dictionary (`indicators_dictionary.py` / `indicators_dictionary_english.py`) defines a Python dict mapping **skill codes** to **skill descriptions**. Skill codes follow the format `<level>.<index>` (e.g., `"3.07"` is the 7th skill at Van Hiele Level 3).

```python
# Example entries
indicators_dict = {
    "2.01": "Uses appropriate vocabulary for geometrical objects and relations ...",
    "3.07": "Identifies shapes belonging to more than one class ...",
    "4.06": "Deduces properties of objects from given or known information ...",
    ...
}
```

The full dictionary contains **33 skills** spanning all five Van Hiele levels (Levels 1–5). It was constructed collaboratively with mathematics education researchers and is used in both classification methods.
