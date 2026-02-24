"""Medical scope guard — validates that agent actions stay within safe boundaries.

Pre-processing step that classifies user input before it reaches the agent.
Uses keyword matching for MVP; designed to be swapped for LLM-based
classification later.
"""

from __future__ import annotations

import re

# ── Category constants ───────────────────────────────────────────────────────

DATA_RETRIEVAL = "DATA_RETRIEVAL"
CLINICAL_SUPPORT = "CLINICAL_SUPPORT"
DIAGNOSIS_REQUEST = "DIAGNOSIS_REQUEST"
TREATMENT_REQUEST = "TREATMENT_REQUEST"
OUT_OF_SCOPE = "OUT_OF_SCOPE"

# ── Configurable keyword lists ───────────────────────────────────────────────
# Each list contains lowercased phrases.  Checked against the lowercased input.
# Order matters: more restrictive categories are checked first so that a query
# like "diagnose drug interaction" is blocked as a diagnosis request rather
# than allowed as clinical support.

DIAGNOSIS_KEYWORDS: list[str] = [
    "diagnose",
    "what disease",
    "what condition",
    "what's wrong with",
    "what is wrong with",
]

TREATMENT_KEYWORDS: list[str] = [
    "prescribe",
    "what should i take",
    "recommend treatment",
    "what medication should",
]

CLINICAL_SUPPORT_KEYWORDS: list[str] = [
    "interaction",
    "interactions",
    "allergy",
    "allergies",
    "allergic",
    "medication",
    "medications",
    "drug",
    "prescription",
]

DATA_RETRIEVAL_KEYWORDS: list[str] = [
    "look up",
    "lookup",
    "find",
    "search",
    "show",
    "list",
    "get",
    "who is",
    "patient",
    "provider",
    "record",
]

# ── Block messages ───────────────────────────────────────────────────────────

BLOCK_MESSAGES: dict[str, str] = {
    DIAGNOSIS_REQUEST: (
        "I'm not able to provide medical diagnoses. Please consult "
        "a qualified healthcare provider for diagnostic assessments."
    ),
    TREATMENT_REQUEST: (
        "I'm not able to recommend treatments or prescribe medications. "
        "Please consult a qualified healthcare provider."
    ),
    OUT_OF_SCOPE: (
        "I'm a healthcare records assistant. I can help you look up "
        "patient information, check drug interactions, and review allergies."
    ),
}

CLINICAL_DISCLAIMER = (
    "\n\n---\n*Disclaimer: This information is for clinical support only. "
    "Please verify with qualified healthcare providers.*"
)

# ── Classification ───────────────────────────────────────────────────────────


def _matches(text: str, keywords: list[str]) -> bool:
    """Return True if *text* contains any of the *keywords* as whole phrases."""
    for kw in keywords:
        # Use word-boundary matching so "list" doesn't match "specialist".
        if re.search(rf"\b{re.escape(kw)}\b", text):
            return True
    return False


def classify_input(user_input: str) -> tuple[str, str | None]:
    """Classify user input into a medical-scope category.

    Args:
        user_input: Raw text from the user.

    Returns:
        A tuple of (category, block_message_or_none).
        ``block_message`` is ``None`` for allowed categories.
    """
    text = user_input.lower().strip()

    # Check blocked categories first (order: most dangerous → least).
    if _matches(text, DIAGNOSIS_KEYWORDS):
        return DIAGNOSIS_REQUEST, BLOCK_MESSAGES[DIAGNOSIS_REQUEST]

    if _matches(text, TREATMENT_KEYWORDS):
        return TREATMENT_REQUEST, BLOCK_MESSAGES[TREATMENT_REQUEST]

    # Allowed categories.
    if _matches(text, CLINICAL_SUPPORT_KEYWORDS):
        return CLINICAL_SUPPORT, None

    if _matches(text, DATA_RETRIEVAL_KEYWORDS):
        return DATA_RETRIEVAL, None

    # Nothing matched → out of scope.
    return OUT_OF_SCOPE, BLOCK_MESSAGES[OUT_OF_SCOPE]


def apply_scope_guard(user_input: str) -> tuple[bool, str | None]:
    """Pre-process a user message and decide whether to allow it through.

    Args:
        user_input: Raw text from the user.

    Returns:
        A tuple of (is_allowed, block_message_if_not_allowed).
        When ``is_allowed`` is ``True``, the second element is ``None``.
    """
    category, block_message = classify_input(user_input)

    if block_message is not None:
        return False, block_message

    return True, None
