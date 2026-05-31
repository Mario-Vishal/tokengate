"""Baseline-vs-ContextPilot token-savings benchmark (CP-024).

Demonstrates the core value: instead of stuffing every retrieved chunk into the prompt
(the "baseline"), ContextPilot ranks/dedups/compresses/budgets and sends far fewer
tokens, with an audit. Uses the deterministic fake models so it runs instantly offline
and focuses on the *token-efficiency* story (model-quality benchmarking is future work).

Run:  uv run python benchmarks/benchmark.py
Writes a markdown summary to benchmarks/output/ (gitignored).
"""

from __future__ import annotations

from pathlib import Path

from contextpilot import ContextBlock, ContextPilot, HeuristicTokenCounter
from contextpilot.models import FakeEmbeddingModel, FakeReranker
from contextpilot.prompts.prompt_builder import build_prompt

_COUNTER = HeuristicTokenCounter()
_QUERY = "Summarize documents about my Cisco job search and list next action items."

# A synthetic, non-private workspace: a few relevant chunks among noise, varied sizes.
_RELEVANT = [
    "Cisco Software Engineer role: build distributed systems; required skills Python, "
    "FastAPI, model evaluation, data pipelines. " * 6,
    "Recruiter email: send your updated resume by Friday and schedule a screening call. "
    "Action items: update resume, reply to recruiter, prepare portfolio. " * 4,
    "Mario resume highlights: Python, FastAPI, AI systems, LanceDB, 2026 job search. " * 5,
    "Interview prep notes for the Cisco networking and systems role. " * 4,
]
_NOISE = [
    "Grocery receipt: bananas, milk, eggs, bread, total amount due. " * 8,
    "Equal opportunity employer statement and benefits boilerplate text. " * 10,
    "Cafeteria menu changes every Friday; remember to water the office plants. " * 6,
    "Software license agreement and terms of service footer text. " * 12,
    "Old newsletter about general AI recruiting industry trends in 2019. " * 7,
    "Random meeting notes about an unrelated marketing campaign. " * 6,
]


def _make_blocks() -> list[ContextBlock]:
    blocks: list[ContextBlock] = []
    for i, text in enumerate(_RELEVANT):
        blocks.append(ContextBlock(content=text, source_id="downloads", block_id=f"rel-{i}"))
    for i, text in enumerate(_NOISE):
        blocks.append(ContextBlock(content=text, source_id="downloads", block_id=f"noise-{i}"))
    return blocks


def _baseline_tokens(query: str, blocks: list[ContextBlock]) -> int:
    """Naive RAG: stuff every candidate block into the prompt."""
    return _COUNTER.count(build_prompt(query, blocks))


def _pilot(strategy: str) -> ContextPilot:
    return ContextPilot(
        max_prompt_tokens=1024,
        strategy=strategy,
        embedding_model=FakeEmbeddingModel(dim=256),
        reranker=FakeReranker(),
        token_counter=_COUNTER,
    )


def main() -> None:
    blocks = _make_blocks()
    baseline = _baseline_tokens(_QUERY, blocks)

    lines = [
        "# ContextPilot benchmark — baseline vs neural (fake models)",
        "",
        f"- Query: `{_QUERY}`",
        f"- Candidate blocks: **{len(blocks)}**  ·  Baseline (stuff-all) prompt tokens: "
        f"**{baseline}**",
        "",
        "| Strategy | Final tokens | Tokens saved | Saved % | Incl | Compr | Dropped |",
        "|----------|-------------:|-------------:|--------:|-----:|------:|--------:|",
    ]
    print(lines[0])
    print(f"baseline (stuff-all) tokens: {baseline} across {len(blocks)} blocks\n")

    for strategy in ("speed", "balanced", "quality", "max_compression"):
        result = _pilot(strategy).optimize(_QUERY, [b.copy() for b in blocks])
        a = result.audit
        vs_baseline = baseline - a.final_prompt_tokens
        pct = (vs_baseline / baseline * 100) if baseline else 0.0
        lines.append(
            f"| {strategy} | {a.final_prompt_tokens} | {vs_baseline} | {pct:.1f}% | "
            f"{a.included_count} | {a.compressed_count} | {a.dropped_count} |"
        )
        print(
            f"{strategy:>16}: final={a.final_prompt_tokens:>5}  "
            f"saved_vs_baseline={vs_baseline:>5} ({pct:5.1f}%)  "
            f"incl={a.included_count} compr={a.compressed_count} drop={a.dropped_count}"
        )

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
