# ContextPilot

**ContextPilot** is a production-grade, neural context-optimization library for LLM applications. Given a user query and a pool of candidate context blocks (retrieved chunks, documents, notes), it decides what to **include, compress, deduplicate, rank, and budget** before a prompt is sent to an LLM — and produces a full **audit report** explaining every decision.

[![python](https://img.shields.io/badge/python-3.12-blue)](https://python.org)
[![version](https://img.shields.io/badge/version-v0.2.0-green)](https://github.com/Mario-Vishal/contextpilot)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## Why

LLM context windows are finite and expensive. The naive approach — stuff every retrieved chunk into the prompt — wastes tokens, degrades answer quality, and gives you zero visibility into *why* the model saw what it saw.

ContextPilot turns context assembly into an explicit, testable, auditable **neural pipeline**:

- Every block passes through embedding-based ranking, cross-encoder reranking, semantic deduplication, extractive compression, and value-per-token budgeting.
- A full audit report shows exactly what was kept, compressed, or dropped — and why.
- No LLM calls in the pipeline. Pure Python, runs offline.

---

## Quickstart

```bash
pip install contextpilot
```

> **First run downloads models.** BGE-M3 (embedder) and BGE-Reranker-v2-m3 (cross-encoder) are fetched from Hugging Face on first use (~1–2 GB total) and cached locally. Requires Python 3.12 and PyTorch.

```python
from contextpilot import ContextPilot, ContextBlock

pilot = ContextPilot(strategy="balanced")   # speed / balanced / quality / max_compression

blocks = [
    ContextBlock(content="FastAPI is a modern Python web framework...", semantic_score=0.85),
    ContextBlock(content="Grocery list: bananas, milk, eggs...",        semantic_score=0.12),
    ContextBlock(content="FastAPI handles async requests efficiently...", semantic_score=0.79),
    # ... up to thousands of candidates
]

result = pilot.optimize(query="How does FastAPI handle async requests?", blocks=blocks)

print(result.final_prompt)        # ready to send to any LLM
print(result.audit.tokens_saved)  # tokens eliminated by the pipeline
print(result.audit.tokens_saved_percent)  # e.g. 74.3 (%)

for decision in result.audit.decisions:
    print(decision.block_id, decision.decision, decision.reason)
# → "abc123" "included"   "high rerank score + within budget"
# → "def456" "dropped"    "semantic duplicate of abc123"
# → "ghi789" "compressed" "relevance-pruned from 312→48 tokens"
```

---

## How it works

```
candidate blocks  ─┐
                   ▼
          1. exact dedup       — remove identical content
          2. BGE-M3 embed      — reuse provided vectors or compute
          3. hybrid rank       — semantic + keyword + recency + source priority
          4. BGE rerank        — cross-encoder scores (query, chunk)
          5. relevance floor   — drop clearly off-topic blocks
          6. semantic dedup    — collapse near-paraphrases (cosine threshold)
          7. extractive compress — keep only query-relevant sentences
          8. MMR selection     — diversity-aware final set
          9. token budget      — value-per-token greedy fit
         10. prompt build      — cacheable sections first, query last
         11. audit             — per-block decisions, per-stage trace, token math
                   │
                   ▼
          OptimizationResult
            .final_prompt          # assembled prompt string
            .included_blocks       # full blocks kept
            .compressed_blocks     # blocks with sentences pruned
            .dropped_blocks        # blocks that didn't make it
            .audit                 # AuditReport with per-stage + per-block detail
```

---

## Requirements

| Requirement | Notes |
|---|---|
| **Python 3.12** | Pinned for ML wheel compatibility |
| **PyTorch** | GPU optional; CPU fallback automatic |
| `sentence-transformers` | Loads `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` |
| NVIDIA GPU (optional) | Speeds up embedding/reranking; not required |

Install with GPU support (CUDA 12.8):

```bash
pip install contextpilot
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

---

## Standalone dashboard (optional)

The `[dashboard]` extra adds a local web UI to explore optimization audit history:

```bash
pip install "contextpilot[dashboard]"
```

To record audits from your app:

```python
from contextpilot.dashboard import AuditStore

store = AuditStore()   # SQLite in ~/.contextpilot/audits.db by default
store.record(session_id="my-session", query=query, result=result)
```

Then launch the dashboard:

```bash
python -m contextpilot.dashboard   # opens http://localhost:8765
```

The dashboard shows per-session, per-query breakdowns: token savings, pipeline funnel (blocks in/out per stage), per-block decisions with content previews, and the full final prompt sent to the LLM.

---

## Strategies

```python
ContextPilot(strategy="speed")           # fast: skip reranker, loose dedup
ContextPilot(strategy="balanced")        # default: full pipeline, balanced weights
ContextPilot(strategy="quality")         # aggressive reranking + tighter dedup
ContextPilot(strategy="max_compression") # maximum token savings
```

Override individual parameters:

```python
from contextpilot.core.config import OptimizerConfig

pilot = ContextPilot(
    config=OptimizerConfig.for_strategy(
        "balanced",
        max_prompt_tokens=8192,
        rerank_top_n=30,
        relevance_floor=0.05,
        semantic_dedup_threshold=0.88,
        mmr_lambda=0.6,
    )
)
```

---

## Bring your own models

```python
from contextpilot.models import EmbeddingModel, Reranker

class MyEmbedder(EmbeddingModel):
    def embed(self, texts: list[str]) -> np.ndarray: ...

class MyReranker(Reranker):
    def rerank(self, query: str, texts: list[str]) -> list[float]: ...

pilot = ContextPilot(embedding_model=MyEmbedder(), reranker=MyReranker())
```

---

## Development

This project uses [**uv**](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Mario-Vishal/contextpilot.git
cd contextpilot
uv sync                     # install all deps (Python 3.12 venv)
uv run pytest               # 195 tests
uv run ruff check src tests # lint
uv run mypy src             # strict type check
```

GPU integration tests (requires real models):

```bash
uv run pytest -m gpu        # downloads BGE models on first run
```

---

## License

[MIT](LICENSE)
