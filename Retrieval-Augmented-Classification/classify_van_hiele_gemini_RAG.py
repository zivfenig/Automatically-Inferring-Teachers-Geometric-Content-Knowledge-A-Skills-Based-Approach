# !pip install -qU google-genai tqdm pandas

import os, re, sys, json
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
from tqdm import tqdm
from google import genai

# ============================================================================
# CONFIGURATION IMPORT
# ============================================================================
# All paths, credentials, and hyperparameters are managed in config_rag.py
from config_rag import *

# Create the main results directory defined in the config
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

<<<<<<< Updated upstream
# Cross-validation structure
FOLDS_DIR = BASE.parent / "Data-and-preprocess" / "HE_Van_Hiele_Dataset" / "folds"
EMB_BASE = BASE / "embeddings_folds" / "HE_embedded_folds"

# Results directory
RESULTS = BASE / "results" / "gemini"
RESULTS.mkdir(parents=True, exist_ok=True)

# Imports (retriever & prompt builders)
sys.path.insert(0, str(BASE / "rag_mechanism"))
sys.path.insert(0, str(BASE / "prompts/He_prompts"))
=======
# Add custom module paths to sys.path
sys.path.insert(0, str(BASE_DIR / "rag_mechanism"))
sys.path.insert(0, str(BASE_DIR / "prompts/HW_prompts"))
>>>>>>> Stashed changes

from retrieve import RAGRetriever, RetrieverConfig
from prompt_only_definitions import build_prompt_only_definitions
from prompt_full_doc_with_indicators import build_prompt_full_doc_with_indicators

# ============================================================================
# INITIALIZE VERTEX AI CLIENT
# ============================================================================
# The client is configured using PROJECT_ID and LOCATION from the config file.
# Ensure you have run 'gcloud auth application-default login' locally.
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def parse_level_and_explanation(text: str) -> Tuple[str, str]:
    """Extract רמה and הסבר fields from Gemini output."""
    if not isinstance(text, str):
        return "", ""
    t = text.strip()
    level, expl = "", ""

    m1 = re.search(r"רמה\s*:\s*([1-5])", t)
    if m1:
        level = m1.group(1)
    m2 = re.search(r"הסבר\s*:\s*(.+)", t, flags=re.DOTALL)
    if m2:
        expl = m2.group(1).strip()

    if not level:
        m = re.search(r"\b([1-5])\b", t)
        if m:
            level = m.group(1)

    if not expl and "\n" in t:
        parts = t.splitlines()
        if len(parts) >= 2:
            expl = parts[1].strip()
    return level, expl


def vertex_generate(system_text: str, user_text: str) -> str:
    """Run Vertex AI generation using settings from the config."""
    resp = client.models.generate_content(
        model=MODEL_ID,
        contents=[{"role": "user", "parts": [{"text": user_text}]}],
        config={
            "system_instruction": [{"text": system_text}],
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature": TEMPERATURE,
        },
    )
    return resp.candidates[0].content.parts[0].text.strip()


def build_prompt_A(retriever, q: str, a: str, k: int, alpha: float) -> Dict[str, Any]:
    """Variant A – Only definitions + RAG."""
    return build_prompt_only_definitions(
        retriever=retriever,
        definitions_path=ONLY_DEFS_TXT,
        question_he=q,
        answer_he=a,
        k=k,
        alpha=alpha,
    )


def build_prompt_B(retriever, q: str, a: str, k: int, alpha: float) -> Dict[str, Any]:
    """Variant B – Operative doc + indicators + RAG."""
    return build_prompt_full_doc_with_indicators(
        retriever=retriever,
        operative_short_path=OPERATIVE_DOC,
        indicators_py_path=INDICATORS_PY,
        question_he=q,
        answer_he=a,
        k=k,
        alpha=alpha,
    )


# ============================================================================
# CORE CLASSIFICATION FUNCTION
# ============================================================================
def classify_van_hiele(
    fold_id: int,
    variant: str = "B",
    k: int = 5,
) -> pd.DataFrame:
    """
    Classify Van Hiele levels for a fold using Gemini.
    
    Args:
        fold_id: Fold number (e.g., 1-5)
        variant: "A" (definitions only) or "B" (full doc + indicators)
        k: Number of retrieved examples
    
    Returns:
        DataFrame with predictions
    """
    if variant not in ["A", "B"]:
        raise ValueError("variant must be 'A' or 'B'")

    test_csv = FOLDS_DIR / f"fold_{fold_id}_test.csv"
    emb_parquet = EMB_BASE_DIR / f"fold_{fold_id}" / f"fold_{fold_id}_train_embeddings.parquet"

    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test set: {test_csv}")
    if not emb_parquet.exists():
        raise FileNotFoundError(f"Missing embeddings: {emb_parquet}")

    # Build retriever using parameters from config
    retriever = RAGRetriever(RetrieverConfig(
        embeddings_path=emb_parquet,
        alpha_answer_weight=ALPHA,
        top_k=TOP_K_FOR_RETRIEVER,
    ))

    # Load test data
    df = pd.read_csv(test_csv).fillna({"question": "", "answer": ""})
    if "question" not in df or "answer" not in df:
        raise ValueError("Missing question/answer columns in test CSV")

    # Choose prompt builder
    BUILD_FN = build_prompt_A if variant == "A" else build_prompt_B

    # Results directory
    out_dir = RESULTS_DIR / f"variant_{variant}" / f"fold_{fold_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"gemini_fold{fold_id}_k{k}.csv"

    # Resume support
    rows_out = []
    processed = set()
    if out_csv.exists() and out_csv.stat().st_size > 0:
        prev = pd.read_csv(out_csv)
        if "test_row_id" in prev:
            processed = set(prev["test_row_id"].tolist())
        rows_out = prev.to_dict("records")

    # Process rows
    to_do = [(i, r["question"], r["answer"]) for i, r in df.iterrows() if i not in processed]
    print(f"[Fold {fold_id} | Variant {variant} | K={k}] → {len(to_do)} rows to process")

    pbar = tqdm(total=len(to_do), desc=f"Gemini {variant} fold {fold_id} K={k}", unit="row")

    for idx, q, a in to_do:
        try:
            prompt = BUILD_FN(retriever, q, a, k=k, alpha=ALPHA)
            system_text, user_text = prompt["system"], prompt["user"]
            
            raw = vertex_generate(system_text, user_text)
            level, expl = parse_level_and_explanation(raw)

            row = {
                "fold": fold_id,
                "variant": variant,
                "k": k,
                "test_row_id": idx,
                "question": q,
                "answer": a,
                "final_decision": df.loc[idx, "final_decision"] if "final_decision" in df else "",
                "final_indicators": df.loc[idx, "final_indicators"] if "final_indicators" in df else "",
                "model_level": level,
                "model_explanation": expl,
                "raw_model_text": raw,
            }
        except Exception as e:
            row = {
                "fold": fold_id,
                "variant": variant,
                "k": k,
                "test_row_id": idx,
                "question": q,
                "answer": a,
                "final_decision": "",
                "final_indicators": "",
                "model_level": "",
                "model_explanation": f"ERROR: {str(e)}",
                "raw_model_text": "",
            }

        rows_out.append(row)
        pbar.update(1)

        # Save every 5 rows
        if len(rows_out) % 5 == 0:
            pd.DataFrame(rows_out).to_csv(out_csv, index=False, encoding="utf-8")

    pbar.close()
    result_df = pd.DataFrame(rows_out)
    result_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Saved → {out_csv}")
    return result_df


# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================
if __name__ == "__main__":
    print("🚀 Starting RAG classification process based on config_rag.py")
    
<<<<<<< Updated upstream
    # Example 2: Classify fold 1 with Variant A, k=3
    # results_a = classify_van_hiele(fold_id=1, variant="A", k=3)
    
    # Example 3: Run all folds with Variant B
    # for fold_id in [1, 2, 3, 4, 5]:
    #     for k in [3, 5, 8, 10]:
    #         classify_van_hiele(fold_id=fold_id, variant="B", k=k)
=======
    # Example 1: Run all folds and K-values for Variant B
    print("\n--- Running Variant B ---")
    for fold_id in FOLDS_TO_RUN:
        for k in K_VALUES_TO_TEST:
            classify_van_hiele(fold_id=fold_id, variant="B", k=k)
            
    # Example 2: Run all folds and K-values for Variant A (optional, uncomment to run)
    # print("\n--- Running Variant A ---")
    # for fold_id in FOLDS_TO_RUN:
    #     for k in K_VALUES_TO_TEST:
    #         classify_van_hiele(fold_id=fold_id, variant="A", k=k)

    print("\n✅ All experiments complete.")
>>>>>>> Stashed changes
