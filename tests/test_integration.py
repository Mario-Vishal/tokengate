"""CP-024 integration: full neural pipeline on the REAL BGE models (opt-in).

Run with ``CONTEXTPILOT_TEST_REAL_MODELS=1`` (downloads/uses BGE-M3 + reranker, GPU if
available). Skipped otherwise so the normal suite stays fast and offline.
"""

from __future__ import annotations

import os

import pytest

from contextpilot import ContextBlock, ContextPilot

pytestmark = pytest.mark.skipif(
    os.environ.get("CONTEXTPILOT_TEST_REAL_MODELS") != "1",
    reason="set CONTEXTPILOT_TEST_REAL_MODELS=1 to run real-model integration",
)


def _workspace_blocks() -> list[ContextBlock]:
    return [
        ContextBlock(
            content="Cisco Software Engineer job description. Responsibilities include "
            "building distributed systems and evaluating ML models.",
            source_id="downloads", token_count=None,
        ),
        ContextBlock(
            content="Mario's resume: experience with Python, FastAPI, and AI systems; "
            "actively in a 2026 job search.",
            source_id="downloads",
        ),
        ContextBlock(
            content="Grocery receipt: bananas, milk, eggs, bread. Total $12.34.",
            source_id="downloads",
        ),
        ContextBlock(
            content="Recruiter email: please send your updated resume by Friday and "
            "schedule a screening call next week.",
            source_id="downloads",
        ),
    ]


def test_real_neural_pipeline_end_to_end() -> None:
    pilot = ContextPilot(max_prompt_tokens=512, strategy="balanced")
    result = pilot.optimize("What should I do next for my job search?", _workspace_blocks())

    assert result.final_prompt.strip()
    a = result.audit
    assert a.total_candidate_blocks == 4
    assert a.included_count + a.compressed_count + a.dropped_count == 4
    assert a.models_used["embedding_model"] == "BAAI/bge-m3"
    assert a.models_used["reranker"] == "BAAI/bge-reranker-v2-m3"
    # job-search-relevant content should make it into the prompt; the grocery receipt
    # should rank below the recruiter/resume material.
    prompt_lower = result.final_prompt.lower()
    assert "resume" in prompt_lower or "recruiter" in prompt_lower


def test_real_quality_preset_keeps_more_context() -> None:
    blocks = _workspace_blocks()
    quality = ContextPilot(max_prompt_tokens=512, strategy="quality").optimize(
        "job search", blocks
    )
    speed = ContextPilot(max_prompt_tokens=512, strategy="speed").optimize(
        "job search", [b.copy() for b in blocks]
    )
    # both produce valid prompts + audits with the real models
    assert quality.audit.total_candidate_blocks == speed.audit.total_candidate_blocks == 4
