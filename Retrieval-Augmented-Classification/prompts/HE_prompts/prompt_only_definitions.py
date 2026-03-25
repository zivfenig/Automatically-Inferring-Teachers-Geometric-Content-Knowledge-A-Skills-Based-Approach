# prompting/prompt_only_definitions.py

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
            f"{i}) שאלה: {q}\n"
            f"   תשובה: {a}\n"
            f"   רמת מומחה (GT): {lvl}\n"
            f"   דמיון (חישוב פנימי): {score:.3f}"
        )
    return "\n\n".join(lines)

def build_prompt_only_definitions(
    retriever,                     # instance of RAGRetriever
    definitions_path: str | Path,  # only_definitions.txt (ללא אינדיקטורים)
    question_he: str,
    answer_he: str,
    k: int = 5,
    alpha: float = 0.8,
    max_q_chars: int = 8000,
    max_a_chars: int = 8000,
) -> Dict[str, Any]:
    """
    Variant A:
      - מערכת: מסמך ההגדרות בלבד (ללא אינדיקטורים)
      - משתמש: Q+A נוכחיים + דוגמאות ICL של Top-K (עם רמת מומחה בלבד)
      - פלט חובה: רמה (ספרה 1–5) + הסבר קצר (2–3 משפטים) המבוסס על ההגדרות בלבד.
    """
    defs_txt = _read_text(Path(definitions_path))

    # Retrieve top-K examples (levels only)
    examples_full = retriever.topk(question_he, answer_he, k=k, alpha=alpha)
    examples_block = _render_examples_levels_only(examples_full)
    # (Keep as metadata if you want) examples_lvl = retriever.to_levels_only(examples_full)

    # Truncate current Q & A for token budget
    q_cur = (question_he or "").strip()[:max_q_chars]
    a_cur = (answer_he or "").strip()[:max_a_chars]

    system = (
        "את/ה אנליסט/ית מומחה/ית לרמות ואן היל בגיאומטריה (Van Hiele Level Analyst Expert).\n"
        "להלן מסמך ההגדרות הרשמי בלבד לרמות 1–5. אין להשתמש בידע חיצוני.\n\n"
        f"{defs_txt}\n\n"
        "הנחיות חשובות:\n"
        "• סווג/י את תשובת המורה לרמת ואן היל (1–5) לפי ההגדרות בלבד.\n"
        "• השתמש/י באופן אקטיבי בדוגמאות הדומות שנשלפו (ICL). ייתכן שיופיעו דוגמאות מכמה רמות שונות; "
        "כאשר הראיות מעורבות, נטה/י לרמת **הרוב** בין ה-K הדוגמאות, כל עוד הדבר עקבי עם ההגדרות.\n"
        "• פורמט פלט מחייב:\n"
        "  רמה: <ספרה בין 1 ל-5>\n"
        "  הסבר: 2–3 משפטים קצרים המבוססים על ההגדרות בלבד (ללא שימוש באינדיקטורים/רשימות).\n"
        "• אין להחזיר טקסט נוסף מעבר לשתי השורות הנ\"ל."
    )

    user = (
        "דוגמה נוכחית לסיווג:\n"
        f"שאלה: {q_cur}\n"
        f"תשובה: {a_cur}\n\n"
        "דוגמאות דומות (עם רמת מומחה):\n"
        f"{examples_block}\n\n"
        "החזר/י כעת בפורמט:\n"
        "רמה: <1–5>\n"
        "הסבר: <2–3 משפטים קצרים>"
    )

    return {"system": system, "user": user}
