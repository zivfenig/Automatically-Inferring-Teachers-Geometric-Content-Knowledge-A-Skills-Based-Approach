from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

def _read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()

def _render_examples_levels_only(examples: List[Dict[str, Any]]) -> str:
    lines = []
    for i, ex in enumerate(examples, 1):
        q = str(ex.get("question", "")).strip()
        a = str(ex.get("answer", "")).strip()
        lvl = ex.get("final_decision", None)
        score = ex.get("score", 0.0)
        lines.append(
            f"{i}) Question: {q}\n"
            f"   Answer: {a}\n"
            f"   Expert Level (GT): {lvl}\n"
            f"   Similarity (Internal Calculation): {score:.3f}"
        )
    return "\n\n".join(lines)

def build_prompt_only_definitions(
    retriever,                     # instance of RAGRetriever
    definitions_path: str | Path,  # only_definitions.txt (without indicators)
    question_en: str,
    answer_en: str,
    k: int = 5,
    alpha: float = 0.8,
    max_q_chars: int = 8000,
    max_a_chars: int = 8000,
) -> Dict[str, Any]:
    """
    Variant A:
      - System: Definitions document only (without indicators)
      - User: Current Q+A + Top-K examples ICL (with expert level only)
      - Required output: Level (digit 1–5) + brief explanation (2–3 sentences) based on definitions only.
    """
    defs_txt = _read_text(Path(definitions_path))

    # Retrieve top-K examples (levels only)
    examples_full = retriever.topk(question_en, answer_en, k=k, alpha=alpha)
    examples_block = _render_examples_levels_only(examples_full)

    # Truncate current Q & A for token budget
    q_cur = (question_en or "").strip()[:max_q_chars]
    a_cur = (answer_en or "").strip()[:max_a_chars]

    system = (
        "You are an expert analyst for Van Hiele levels in geometry (Van Hiele Level Analyst Expert).\n"
        "Below is the official definitions document only for levels 1–5. Do not use external knowledge.\n\n"
        f"{defs_txt}\n\n"
        "Important Guidelines:\n"
        "• Classify the teacher's answer to a Van Hiele level (1–5) based on the definitions only.\n"
        "• Actively use the similar examples retrieved (ICL). Examples from multiple different levels may appear; "
        "when evidence is mixed, lean towards the **majority level** among the K examples, as long as it is consistent with the definitions.\n"
        "• Required output format:\n"
        "  Level: <digit between 1 and 5>\n"
        "  Explanation: 2–3 short sentences based on the definitions only (without using indicators/lists).\n"
        "• Do not return any text beyond the two lines above."
    )

    user = (
        "Current example to classify:\n"
        f"Question: {q_cur}\n"
        f"Answer: {a_cur}\n\n"
        "Similar examples (with expert level):\n"
        f"{examples_block}\n\n"
        "Now return in the format:\n"
        "Level: <1–5>\n"
        "Explanation: <2–3 short sentences>"
    )

    return {"system": system, "user": user}