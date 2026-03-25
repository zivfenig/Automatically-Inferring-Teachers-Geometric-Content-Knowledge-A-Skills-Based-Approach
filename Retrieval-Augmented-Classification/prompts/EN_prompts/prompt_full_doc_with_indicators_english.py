from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import importlib.util
import json

def _read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()

def _import_indicators_dict(py_path: str | Path) -> Tuple[dict, str]:
    """
    Loads a Python file that defines an indicators dictionary.
    Expected: a top-level variable like INDICATORS / INDICATORS_DICT / indicator_dict.
    Returns (dict_object, variable_name_used).
    """
    py_path = Path(py_path)
    spec = importlib.util.spec_from_file_location("inds_mod", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    cand_names = ["INDICATORS", "INDICATORS_DICT", "indicators_dict", "INDICATOR_DICT", "indicator_dict"]
    for name in cand_names:
        if hasattr(mod, name):
            obj = getattr(mod, name)
            if isinstance(obj, dict):
                return obj, name
    # Fallback: first dict found
    for k, v in mod.__dict__.items():
        if isinstance(v, dict):
            return v, k
    raise ValueError(f"No indicators dictionary found in file: {py_path}")

def _format_indicators_maybe_json(inds_raw: Any) -> str:
    """
    Accept various shapes (str JSON, dict, list of ids) and format as compact text.
    """
    if inds_raw is None:
        return ""
    if isinstance(inds_raw, dict):
        items = []
        for k, v in inds_raw.items():
            k = str(k).strip()
            if not k:
                continue
            vv = str(v).strip()
            items.append(f"{k}: {vv}")
        return "; ".join(items)
    if isinstance(inds_raw, list):
        return ", ".join(str(x) for x in inds_raw)
    if isinstance(inds_raw, str):
        s = inds_raw.strip()
        if not s:
            return ""
        try:
            obj = json.loads(s)
            return _format_indicators_maybe_json(obj)
        except Exception:
            return s
    return str(inds_raw)

def _render_examples_with_inds(examples: List[Dict[str, Any]]) -> str:
    """
    Format top-K examples including expert indicators.
    """
    lines = []
    for i, ex in enumerate(examples, 1):
        q = str(ex.get("question", "")).strip()
        a = str(ex.get("answer", "")).strip()
        lvl = ex.get("final_decision", None)
        inds_fmt = _format_indicators_maybe_json(ex.get("final_indicators", ""))
        score = ex.get("score", 0.0)
        lines.append(
            f"{i}) Question: {q}\n"
            f"   Answer: {a}\n"
            f"   Expert Level (GT): {lvl}\n"
            f"   Indicators (GT): {inds_fmt}\n"
            f"   Similarity (Internal Calculation): {score:.3f}"
        )
    return "\n\n".join(lines)

def build_prompt_full_doc_with_indicators(
    retriever,                         # instance of RAGRetriever
    operative_short_path: str | Path,  # Operative_doc_short_version.txt (definitions + indicators)
    indicators_py_path: str | Path,    # indicators_dictionary.py (includes all indicators)
    question_en: str,
    answer_en: str,
    k: int = 5,
    alpha: float = 0.8,
    max_q_chars: int = 8000,
    max_a_chars: int = 8000,
) -> Dict[str, Any]:
    """
    Variant B:
      - System: Short operative document (includes definitions + indicators)
      - User: Current Q+A + ICL examples with expert level + indicators
      - Additionally: Load indicators dictionary as terminology guidance (for display only).
      - Goal: Return a single digit 1..5 + brief explanation (2–3 sentences) based on definitions and indicators.
    """
    op_txt = _read_text(Path(operative_short_path))
    indicators_dict, var_name = _import_indicators_dict(indicators_py_path)

    indicators_preview_json = json.dumps(indicators_dict, ensure_ascii=False)

    # Retrieve top-K examples (with indicators)
    examples_full = retriever.topk(question_en, answer_en, k=k, alpha=alpha)
    examples_lvl_inds = retriever.to_levels_and_indicators(examples_full)
    examples_block = _render_examples_with_inds(examples_full)

    # Truncate current Q & A for token budget
    q_cur = (question_en or "").strip()[:max_q_chars]
    a_cur = (answer_en or "").strip()[:max_a_chars]

    system = (
        "You are an expert analyst for Van Hiele levels in geometry (Van Hiele Level Analyst Expert).\n"
        "Below is a short operative document containing definitions for levels 1–5 and indicators for level assignment. "
        "Base your analysis only on the document below and the similar examples—do not use external knowledge.\n\n"
        f"{op_txt}\n\n"
        "Indicators Dictionary:\n"
        f"{indicators_preview_json}\n\n"
        "Important Guidelines:\n"
        "• Classify the teacher's answer to a Van Hiele level (1–5), based on the definitions and indicators in the document.\n"
        "• Actively use the similar examples retrieved (ICL). Examples from multiple different levels may appear; "
        "when evidence is mixed, lean towards the **majority level** among the K examples, as long as it is consistent with the definitions and indicators.\n"
        "Note: If, for example, most of the retrieved examples belong to level 5, there is a higher probability that the current classification is also 5—provided that the definitions and indicators match the content.\n"
        "• Emphasize the semantic content of the answer, then use the question for context.\n"
        "• Required output format:\n"
        "  Line 1: Level: <digit between 1 and 5>\n"
        "  Line 2: Explanation: 2–3 short sentences justifying the choice based on the relevant definitions and indicators (do not return an indicators list, but rather a brief textual formulation).\n"
        "• Do not return any text beyond the two lines above."
    )

    user = (
        "Current example to classify:\n"
        f"Question: {q_cur}\n"
        f"Answer: {a_cur}\n\n"
        "Similar examples (with expert level + indicators):\n"
        f"{examples_block}\n\n"
        "Now return the output in the required format:\n"
        "Level: <digit between 1 and 5>\n"
        "Explanation: <2–3 short sentences based on the definitions and indicators>"
    )

    return {
        "system": system,
        "user": user,
        "examples": examples_lvl_inds,
        "meta": {
            "k": k,
            "alpha": alpha,
            "variant": "full_doc_with_indicators",
            "indicators_var": var_name,
        },
    }