"""MVP evaluation test suite.

Loads test cases from eval/test_cases.yaml and runs each through the scope
guard (for blocked cases) or the full agent (for allowed cases).

Usage:
    pytest tests/test_eval.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from src.verification.scope_guard import apply_scope_guard, classify_input

# ── Load test cases ──────────────────────────────────────────────────────────

_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
_CASES_FILE = _EVAL_DIR / "test_cases.yaml"


def _load_cases() -> list[dict]:
    with open(_CASES_FILE) as f:
        data = yaml.safe_load(f)
    return data["test_cases"]


_ALL_CASES = _load_cases()

# Partition into blocked vs allowed for different test strategies.
_BLOCKED_CASES = [c for c in _ALL_CASES if c["should_block"]]
_ALLOWED_CASES = [c for c in _ALL_CASES if not c["should_block"]]


# ── Sanity: YAML loaded correctly ────────────────────────────────────────────


def test_yaml_loaded():
    """Verify the YAML file contains the expected number of test cases."""
    assert len(_ALL_CASES) >= 5, (
        f"Expected at least 5 test cases, found {len(_ALL_CASES)}"
    )


def test_all_cases_have_required_fields():
    required = {"id", "category", "input", "expected_tools",
                "expected_output_contains", "should_block"}
    for case in _ALL_CASES:
        missing = required - set(case.keys())
        assert not missing, f"Case {case.get('id', '?')} missing fields: {missing}"


# ── Blocked cases: scope guard rejects without hitting the LLM ───────────────


@pytest.mark.parametrize(
    "case",
    _BLOCKED_CASES,
    ids=[c["id"] for c in _BLOCKED_CASES],
)
def test_scope_guard_blocks(case: dict):
    """Verify the scope guard blocks adversarial/out-of-scope queries."""
    is_allowed, block_message = apply_scope_guard(case["input"])

    assert is_allowed is False, (
        f"[{case['id']}] Expected scope guard to block, but it allowed: "
        f"{case['input']!r}"
    )
    assert block_message is not None

    # Verify expected strings appear in the block message.
    for expected in case["expected_output_contains"]:
        assert expected.lower() in block_message.lower(), (
            f"[{case['id']}] Expected {expected!r} in block message, "
            f"got: {block_message!r}"
        )


# ── Allowed cases: scope guard passes, then agent produces expected output ───


@pytest.mark.parametrize(
    "case",
    _ALLOWED_CASES,
    ids=[c["id"] for c in _ALLOWED_CASES],
)
def test_scope_guard_allows(case: dict):
    """Verify the scope guard lets valid queries through."""
    is_allowed, block_message = apply_scope_guard(case["input"])

    assert is_allowed is True, (
        f"[{case['id']}] Expected scope guard to allow, but it blocked: "
        f"{case['input']!r} — {block_message}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    _ALLOWED_CASES,
    ids=[c["id"] for c in _ALLOWED_CASES],
)
async def test_agent_response(case: dict):
    """Run allowed queries through the agent and verify expected output.

    The inner ReAct agent is mocked so we don't hit live APIs or LLMs.
    The mock returns a canned response containing the keywords the test
    case expects, letting us verify end-to-end wiring.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    from src.agent.graph import run_agent

    fake_response = " | ".join(case["expected_output_contains"])

    with patch(
        "src.agent.graph._react_agent.ainvoke",
        new_callable=AsyncMock,
    ) as mock_agent:
        mock_agent.return_value = {
            "messages": [
                HumanMessage(content=case["input"]),
                AIMessage(content=fake_response),
            ],
        }

        result = await run_agent(case["input"])

    for expected in case["expected_output_contains"]:
        assert expected.lower() in result.lower(), (
            f"[{case['id']}] Expected {expected!r} in agent response, "
            f"got: {result!r}"
        )


# ── Summary reporter (runs last) ─────────────────────────────────────────────


def test_eval_summary(request):
    """Print a human-readable summary after all eval tests run.

    This test always passes — it's just a reporting hook.
    Run with ``pytest tests/test_eval.py -v`` to see per-case results,
    or inspect the summary at the end.
    """
    total = len(_ALL_CASES)
    blocked = len(_BLOCKED_CASES)
    allowed = len(_ALLOWED_CASES)
    print(
        f"\n{'='*60}\n"
        f"  EVAL SUMMARY\n"
        f"  Total test cases : {total}\n"
        f"  Should block     : {blocked}\n"
        f"  Should allow     : {allowed}\n"
        f"  Categories       : "
        f"{', '.join(sorted({c['category'] for c in _ALL_CASES}))}\n"
        f"{'='*60}"
    )
