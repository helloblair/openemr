"""Tests for the medical scope guard."""

import pytest

from src.verification.scope_guard import (
    BLOCK_MESSAGES,
    CLINICAL_SUPPORT,
    DATA_RETRIEVAL,
    DIAGNOSIS_REQUEST,
    OUT_OF_SCOPE,
    TREATMENT_REQUEST,
    apply_scope_guard,
    classify_input,
)


# ── DATA_RETRIEVAL (allowed) ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Look up patient John Smith",
        "Find me the record for Jane Doe",
        "Search for Dr. Adams",
        "Show me the patient list",
        "Get the provider schedule",
        "Who is patient 12345?",
        "Can you list all patients?",
    ],
)
def test_data_retrieval_allowed(text):
    category, block = classify_input(text)
    assert category == DATA_RETRIEVAL
    assert block is None

    allowed, msg = apply_scope_guard(text)
    assert allowed is True
    assert msg is None


# ── CLINICAL_SUPPORT (allowed) ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Check drug interaction between aspirin and warfarin",
        "What are the allergy records for this patient?",
        "Does this patient have any allergies?",
        "Review the medication list",
        "Is there an interaction with this drug?",
        "Check this prescription for issues",
        "Is the patient allergic to penicillin?",
    ],
)
def test_clinical_support_allowed(text):
    category, block = classify_input(text)
    assert category == CLINICAL_SUPPORT
    assert block is None

    allowed, msg = apply_scope_guard(text)
    assert allowed is True
    assert msg is None


# ── DIAGNOSIS_REQUEST (blocked) ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Diagnose this patient's symptoms",
        "What disease does this patient have?",
        "What condition is causing the pain?",
        "What's wrong with this patient?",
        "Can you diagnose the rash?",
    ],
)
def test_diagnosis_blocked(text):
    category, block = classify_input(text)
    assert category == DIAGNOSIS_REQUEST
    assert block == BLOCK_MESSAGES[DIAGNOSIS_REQUEST]

    allowed, msg = apply_scope_guard(text)
    assert allowed is False
    assert msg == BLOCK_MESSAGES[DIAGNOSIS_REQUEST]


# ── TREATMENT_REQUEST (blocked) ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Prescribe something for the headache",
        "What should I take for a cold?",
        "Recommend treatment for this condition",
        "What medication should the patient be on?",
    ],
)
def test_treatment_blocked(text):
    category, block = classify_input(text)
    assert category == TREATMENT_REQUEST
    assert block == BLOCK_MESSAGES[TREATMENT_REQUEST]

    allowed, msg = apply_scope_guard(text)
    assert allowed is False
    assert msg == BLOCK_MESSAGES[TREATMENT_REQUEST]


# ── OUT_OF_SCOPE (blocked) ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Write me a poem about cats",
        "What's the weather today?",
        "Tell me a joke",
        "How do I cook pasta?",
    ],
)
def test_out_of_scope_blocked(text):
    category, block = classify_input(text)
    assert category == OUT_OF_SCOPE
    assert block == BLOCK_MESSAGES[OUT_OF_SCOPE]

    allowed, msg = apply_scope_guard(text)
    assert allowed is False
    assert msg == BLOCK_MESSAGES[OUT_OF_SCOPE]


# ── Priority: blocked categories take precedence ────────────────────────────


def test_diagnosis_overrides_clinical():
    """A query mentioning both diagnosis and clinical keywords is blocked."""
    text = "Diagnose the drug interaction"
    category, block = classify_input(text)
    assert category == DIAGNOSIS_REQUEST
    assert block is not None


def test_treatment_overrides_data():
    """A query mentioning both treatment and data keywords is blocked."""
    text = "Prescribe medication and show patient record"
    category, block = classify_input(text)
    assert category == TREATMENT_REQUEST
    assert block is not None


# ── Case insensitivity ──────────────────────────────────────────────────────


def test_case_insensitive():
    category, _ = classify_input("LOOK UP PATIENT")
    assert category == DATA_RETRIEVAL

    category, block = classify_input("DIAGNOSE the issue")
    assert category == DIAGNOSIS_REQUEST
    assert block is not None


# ── Word-boundary matching ──────────────────────────────────────────────────


def test_word_boundary_no_false_positive():
    """'list' in 'specialist' should not trigger DATA_RETRIEVAL."""
    text = "I need a specialist recommendation"
    category, _ = classify_input(text)
    # "specialist" contains "list" but word boundary prevents matching.
    # No data retrieval keyword matches, so this falls to OUT_OF_SCOPE.
    assert category == OUT_OF_SCOPE
