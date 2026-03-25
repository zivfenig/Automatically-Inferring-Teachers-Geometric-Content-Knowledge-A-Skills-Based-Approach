# File: src/rag/retrieve.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# ----------------------------
# Config dataclass
# ----------------------------
@dataclass
class RetrieverConfig:
    embeddings_path: Path
    model_name: str = "intfloat/multilingual-e5-base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    alpha_answer_weight: float = 0.8  # 80% answer, 20% question
    top_k: int = 5


# ----------------------------
# Helper functions
# ----------------------------
def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    return "" if s == "nan" else s

def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def _as_json(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return "[]"


# ----------------------------
# Retriever class
# ----------------------------
class RAGRetriever:
    """
    Loads RAG embeddings (question & answer vectors) and retrieves top-K examples
    for a new Hebrew Q+A using an answer-heavy similarity (default 80% answer).
    """
    def __init__(self, cfg: RetrieverConfig):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name, device=cfg.device)
        self.model.max_seq_length = 512

        # Load embeddings parquet
        df = pd.read_parquet(cfg.embeddings_path)
        required_cols = ["id","question","answer","final_decision","final_indicators","q_vec","a_vec"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Embeddings file missing columns: {missing}")

        # Keep a copy (for metadata extraction later)
        self.df = df.reset_index(drop=True)

        # Convert to numpy for fast math
        self.Q = np.vstack(self.df["q_vec"].to_list()).astype(np.float32)  # [N, D]
        self.A = np.vstack(self.df["a_vec"].to_list()).astype(np.float32)  # [N, D]

        # Most SBERT/E5 vectors are already unit norm; still normalize defensively.
        self.Q = _l2_normalize(self.Q)
        self.A = _l2_normalize(self.A)

        # Precompute the **corpus mixed** vectors for the configured alpha (answer-heavy)
        a = float(cfg.alpha_answer_weight)
        self.corpus_mix = _l2_normalize(a * self.A + (1.0 - a) * self.Q)  # [N, D]

    # --------- Embedding for NEW query (Hebrew) ---------
    def _embed_query_pair(self, question_he: str, answer_he: str) -> Tuple[np.ndarray, np.ndarray]:
        # E5 best practice: "query: " prefix for queries
        q_vec = self.model.encode([f"query: { _safe_text(question_he) }"],
                                  convert_to_numpy=True,
                                  normalize_embeddings=True)[0]
        a_vec = self.model.encode([f"query: { _safe_text(answer_he) }"],
                                  convert_to_numpy=True,
                                  normalize_embeddings=True)[0]
        return q_vec.astype(np.float32), a_vec.astype(np.float32)

    def _build_weighted_query(self, q_vec: np.ndarray, a_vec: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        if alpha is None:
            alpha = self.cfg.alpha_answer_weight
        v = (alpha * a_vec) + ((1.0 - alpha) * q_vec)
        v = v / (np.linalg.norm(v) + 1e-12)
        return v.astype(np.float32)

    # --------- Main retrieval ---------
    def topk(
        self,
        question_he: str,
        answer_he: str,
        k: Optional[int] = None,
        alpha: Optional[float] = None,
        exclude_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns top-K most similar examples.
        Each item includes: id, question, answer, final_decision, final_indicators, score
        """
        if k is None:
            k = self.cfg.top_k

        q_vec, a_vec = self._embed_query_pair(question_he, answer_he)
        q_mix = self._build_weighted_query(q_vec, a_vec, alpha=alpha)

        # cosine since all are unit-norm → dot product
        sims = self.corpus_mix @ q_mix  # [N]

        # Exclude specified ids if needed (e.g., same row)
        if exclude_ids:
            # Build a mask of allowed indices
            mask = ~self.df["id"].isin(exclude_ids).to_numpy()
            sims = np.where(mask, sims, -1e9)

        # Argpartition for top-k, then sort
        k = int(min(k, sims.shape[0]))
        top_idx = np.argpartition(-sims, k-1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results: List[Dict[str, Any]] = []
        for idx in top_idx:
            row = self.df.iloc[int(idx)]
            results.append({
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "final_decision": int(row["final_decision"]) if pd.notna(row["final_decision"]) else None,
                "final_indicators": row["final_indicators"],  # keep as-is (string or JSON)
                "score": float(sims[idx]),
            })
        return results

    # --------- Convenience formatters for prompt builders ---------
    @staticmethod
    def to_levels_only(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        For prompts that need only the expert final level (Variant A).
        """
        out = []
        for ex in examples:
            out.append({
                "question": ex["question"],
                "answer": ex["answer"],
                "level": ex["final_decision"],
                "score": ex["score"],
            })
        return out

    @staticmethod
    def to_levels_and_indicators(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        For prompts that include both final level and indicators (Variant B).
        """
        out = []
        for ex in examples:
            out.append({
                "question": ex["question"],
                "answer": ex["answer"],
                "level": ex["final_decision"],
                "indicators": ex["final_indicators"],  # raw string/JSON as stored
                "score": ex["score"],
            })
        return out
