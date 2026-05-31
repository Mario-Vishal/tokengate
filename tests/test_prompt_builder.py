"""CP-011 tests: prompt builder ordering and format."""

from __future__ import annotations

from contextpilot import ContextBlock
from contextpilot.prompts.prompt_builder import build_prompt


def test_query_always_present() -> None:
    prompt = build_prompt("what is X?", [])
    assert prompt == "Question: what is X?"


def test_cacheable_ordered_before_context() -> None:
    blocks = [
        ContextBlock(content="DOC CHUNK", source_id="f1"),
        ContextBlock(content="SYSTEM RULES", cacheable=True),
    ]
    prompt = build_prompt("q", blocks)
    assert prompt.index("SYSTEM RULES") < prompt.index("Context:")
    assert prompt.index("Context:") < prompt.index("DOC CHUNK")
    assert prompt.index("DOC CHUNK") < prompt.index("Question: q")


def test_source_annotation_present_when_available() -> None:
    prompt = build_prompt("q", [ContextBlock(content="text", source_id="file:resume.pdf")])
    assert "(source: file:resume.pdf)" in prompt


def test_no_source_annotation_when_absent() -> None:
    prompt = build_prompt("q", [ContextBlock(content="text")])
    assert "(source:" not in prompt


def test_context_blocks_numbered_in_order() -> None:
    blocks = [
        ContextBlock(content="first", block_id="1"),
        ContextBlock(content="second", block_id="2"),
    ]
    prompt = build_prompt("q", blocks)
    assert "[1]" in prompt and "[2]" in prompt
    assert prompt.index("[1]") < prompt.index("[2]")
    assert prompt.index("first") < prompt.index("second")


def test_deterministic() -> None:
    blocks = [
        ContextBlock(content="SYS", cacheable=True),
        ContextBlock(content="ctx", source_id="s"),
    ]
    assert build_prompt("q", blocks) == build_prompt("q", blocks)


def test_only_cacheable_no_context_section() -> None:
    prompt = build_prompt("q", [ContextBlock(content="SYS ONLY", cacheable=True)])
    assert "Context:" not in prompt
    assert "SYS ONLY" in prompt
    assert prompt.endswith("Question: q")


def test_custom_labels() -> None:
    prompt = build_prompt(
        "q", [ContextBlock(content="c")], context_header="Sources", query_label="Query"
    )
    assert "Sources:" in prompt
    assert "Query: q" in prompt
