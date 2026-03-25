# File: prompting/prompt_full_doc_with_indicators.py
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
    raise ValueError(f"לא נמצא מילון אינדיקטורים בקובץ: {py_path}")

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
            f"{i}) שאלה: {q}\n"
            f"   תשובה: {a}\n"
            f"   רמת מומחה (GT): {lvl}\n"
            f"   אינדיקטורים (GT): {inds_fmt}\n"
            f"   דמיון (חישוב פנימי): {score:.3f}"
        )
    return "\n\n".join(lines)

def build_prompt_full_doc_with_indicators(
    retriever,                         # instance of RAGRetriever
    operative_short_path: str | Path,  # Operative_doc_short_version.txt (הגדרות + אינדיקטורים)
    indicators_py_path: str | Path,    # indicators_dictionary.py (כולל כל האינדיקטורים)
    question_he: str,
    answer_he: str,
    k: int = 5,
    alpha: float = 0.8,
    max_q_chars: int = 8000,
    max_a_chars: int = 8000,
) -> Dict[str, Any]:
    """
    Variant B:
      - מערכת: מסמך אופרטיבי מקוצר (כולל הגדרות + אינדיקטורים)
      - משתמש: Q+A נוכחיים + דוגמאות ICL עם רמת מומחה + אינדיקטורים
      - בנוסף: טעינת מילון האינדיקטורים כהכוונה טרמינולוגית (לתצוגה בלבד).
      - מטרה: החזר ספרה אחת בלבד 1..5 + הסבר קצר (2–3 משפטים) המבוסס על ההגדרות והאינדיקטורים.
    """
    op_txt = _read_text(Path(operative_short_path))
    indicators_dict, var_name = _import_indicators_dict(indicators_py_path)

    indicators_preview_json = json.dumps(indicators_dict, ensure_ascii=False)

    # Retrieve top-K examples (with indicators)
    examples_full = retriever.topk(question_he, answer_he, k=k, alpha=alpha)
    examples_lvl_inds = retriever.to_levels_and_indicators(examples_full)
    examples_block = _render_examples_with_inds(examples_full)

    # Truncate current Q & A for token budget
    q_cur = (question_he or "").strip()[:max_q_chars]
    a_cur = (answer_he or "").strip()[:max_a_chars]

    system = (
        "את/ה אנליסט/ית מומחה/ית לרמות ואן היל בגיאומטריה (Van Hiele Level Analyst Expert).\n"
        "להלן מסמך אופרטיבי מקוצר הכולל הגדרות לרמות 1–5 וכן אינדיקטורים לשיוך רמה. "
        "הסתמך/י אך ורק על המסמך להלן והדוגמאות הדומות—אין להשתמש בידע חיצוני.\n\n"
        f"{op_txt}\n\n"
        "מילון אינדיקטורים:\n"
        f"{indicators_preview_json}\n\n"
        "הנחיות חשובות:\n"
        "• סווג/י את תשובת המורה לרמת ואן היל (1–5), בהתבסס על ההגדרות והאינדיקטורים במסמך.\n"
        "• השתמש/י באופן אקטיבי בדוגמאות הדומות שנשלפו (ICL). ייתכן שיופיעו דוגמאות מכמה רמות שונות; "
        "כאשר הראיות מעורבות, נטה/י לכיוון רמת **הרוב** מבין ה-K הדוגמאות, כל עוד הדבר עקבי עם ההגדרות והאינדיקטורים.\n"
        "שים/י לב: אם לדוגמה רוב הדוגמאות שייכות לרמה 5, יש הסתברות גבוהה יותר שגם הסיווג הנוכחי הוא 5—בתנאי שההגדרות והאינדיקטורים תואמים את התוכן.\n"
        "• הדגש/י את התוכן הסמנטי של התשובה, ולאחר מכן את השאלה להקשר.\n"
        "• פורמט פלט מחייב:\n"
        "  שורה 1: רמה: <ספרה בין 1 ל-5>\n"
        "  שורה 2: הסבר: 2–3 משפטים קצרים המנמקים את הבחירה על בסיס ההגדרות והאינדיקטורים הרלוונטיים (אין להחזיר רשימת אינדיקטורים, אלא ניסוח טקסטואלי קצר).\n"
        "• אין להחזיר טקסט נוסף מעבר לשתי השורות הנ\"ל."
    )

    user = (
        "דוגמה נוכחית לסיווג:\n"
        f"שאלה: {q_cur}\n"
        f"תשובה: {a_cur}\n\n"
        "דוגמאות דומות (עם רמת מומחה + אינדיקטורים):\n"
        f"{examples_block}\n\n"
        "החזר/י כעת את הפלט בפורמט הנדרש:\n"
        "רמה: <ספרה בין 1 ל-5>\n"
        "הסבר: <2–3 משפטים קצרים על בסיס ההגדרות והאינדיקטורים>"
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
