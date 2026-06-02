<div align="center">

# TokenGate

**Neural context optimization for LLM applications.**

Given a query and a pool of retrieved chunks, TokenGate decides what to keep,
compress, and drop before the prompt reaches the LLM — and records every
decision in a full audit report.

[![Python](https://img.shields.io/badge/Python-3.12-3776ab?logo=python&logoColor=white)](https://python.org)
[![Version](https://img.shields.io/badge/version-v0.2.0-22c55e)](https://github.com/Mario-Vishal/tokengate)
[![Tests](https://img.shields.io/badge/tests-195%20passing-22c55e)](https://github.com/Mario-Vishal/tokengate)
[![Typed](https://img.shields.io/badge/mypy-strict-3b82f6)](https://mypy.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT-94a3b8)](LICENSE)

</div>

---

## The problem

The default approach in RAG is to retrieve the top-k chunks and stuff them all into the prompt. That works until it doesn't: token limits blow up, irrelevant passages dilute the answer, and you have no idea what the model actually saw.

TokenGate replaces the stuffing step with an 11-stage neural pipeline. Every candidate goes through embedding-based ranking, cross-encoder reranking, semantic deduplication, extractive compression, and value-per-token budgeting. The model gets a tighter, more relevant prompt. You get a full audit of every decision made along the way.

No LLM calls inside the pipeline. Pure Python. Runs offline.

---

## Install

```bash
pip install git+https://github.com/Mario-Vishal/tokengate.git
```

> **Models are downloaded on first use.** BGE-M3 (embedder) and BGE-Reranker-v2-m3 (cross-encoder) are fetched from Hugging Face automatically and cached locally (~1-2 GB total). Requires Python 3.12 and PyTorch.

With the optional audit dashboard:

```bash
pip install "git+https://github.com/Mario-Vishal/tokengate.git#egg=tokengate[dashboard]"
```

---

## Quickstart

```python
from tokengate import TokenGate, TokenBlock

gate = TokenGate(strategy="balanced")

blocks = [
    TokenBlock(content="FastAPI is a modern Python web framework...", semantic_score=0.85),
    TokenBlock(content="Grocery list: bananas, milk, eggs...",        semantic_score=0.12),
    TokenBlock(content="FastAPI handles async requests efficiently...", semantic_score=0.79),
    # up to thousands of candidates
]

result = gate.optimize(
    query="How does FastAPI handle async requests?",
    blocks=blocks,
)

print(result.final_prompt)               # send this to any LLM
print(result.audit.tokens_saved)         # tokens cut by the pipeline
print(result.audit.tokens_saved_percent) # e.g. 74.3

for d in result.audit.decisions:
    print(d.block_id, d.decision, d.reason)
# abc123  included    high rerank score, within budget
# def456  dropped     semantic duplicate of abc123
# ghi789  compressed  relevance-pruned from 312 to 48 tokens
```

---

## Pipeline

```
  candidate blocks
        |
        v
   1.  exact dedup          remove byte-identical content
   2.  BGE-M3 embed         reuse stored vectors or compute fresh ones
   3.  hybrid rank          semantic + keyword + recency + source priority
   4.  BGE rerank           cross-encoder score per (query, chunk) pair
   5.  relevance floor      discard clearly off-topic blocks early
   6.  semantic dedup       collapse near-paraphrases by cosine threshold
   7.  extractive compress  keep only query-relevant sentences per block
   8.  MMR selection        diversity-aware greedy final set
   9.  token budget         value-per-token fit within the context window
  10.  prompt build         stable/cacheable sections first, query last
  11.  audit                per-block decisions, per-stage trace, token math
        |
        v
   OptimizationResult
     .final_prompt        assembled prompt string, ready for any LLM
     .included_blocks     blocks kept in full
     .compressed_blocks   blocks with irrelevant sentences pruned
     .dropped_blocks      blocks that did not make it through
     .audit               full AuditReport with per-stage and per-block detail
```

---

## Strategies

| Strategy | What it does |
|---|---|
| `speed` | Skips the cross-encoder, loose dedup thresholds. Fastest. |
| `balanced` | Full pipeline with balanced weights. Default. |
| `quality` | Aggressive reranking, tight semantic dedup, higher relevance floor. |
| `max_compression` | Maximum token reduction, relevance-driven sentence pruning. |

```python
gate = TokenGate(strategy="quality")
```

Fine-tune individual parameters:

```python
from tokengate.core.config import OptimizerConfig

gate = TokenGate(
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

TokenGate ships with BGE-M3 and BGE-Reranker-v2-m3 as defaults. Swap them out by implementing two small protocols:

```python
from tokengate.models import EmbeddingModel, Reranker
import numpy as np

class MyEmbedder(EmbeddingModel):
    def embed(self, texts: list[str]) -> np.ndarray: ...

class MyReranker(Reranker):
    def rerank(self, query: str, texts: list[str]) -> list[float]: ...

gate = TokenGate(embedding_model=MyEmbedder(), reranker=MyReranker())
```

---

## Audit dashboard

The `[dashboard]` extra adds a local web UI for exploring audit history across sessions.

```python
from tokengate.dashboard import AuditStore

store = AuditStore()  # persists to ~/.tokengate/audits.db
store.record(session_id="session-1", query=query, result=result)
```

```bash
python -m tokengate.dashboard  # opens http://localhost:8765
```

The dashboard shows token savings, a per-stage pipeline funnel, per-block decisions with content previews, and the full final prompt sent to the LLM.

---

## Requirements

| | |
|---|---|
| Python 3.12 | Pinned for ML wheel compatibility |
| PyTorch | GPU optional, CPU fallback is automatic |
| sentence-transformers | Loads BGE-M3 and BGE-Reranker-v2-m3 |
| NVIDIA GPU | Optional. Speeds up embedding and reranking. |

GPU install (CUDA 12.8):

```bash
pip install git+https://github.com/Mario-Vishal/tokengate.git
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

---

## Development

```bash
git clone https://github.com/Mario-Vishal/tokengate.git
cd tokengate
uv sync                      # Python 3.12 venv + all deps
uv run pytest                # 195 tests
uv run ruff check src tests  # lint
uv run mypy src              # strict type checking
```

GPU integration tests (downloads real models on first run):

```bash
uv run pytest -m gpu
```

---

## License

[MIT](LICENSE)
