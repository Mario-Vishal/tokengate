"""LLMLingua-2 compressor backend (CP-031).

Wraps Microsoft's LLMLingua-2 — a small pretrained token-classification transformer that
labels each token keep/drop by learned **information density** (distilled from GPT-4). Unlike
relevance scoring (which can't tell that a bare ``$154.61`` answers a query), it keeps
high-information tokens — numbers, dates, entities — and drops filler, so it works on prose
and on structureless blobs where sentence-extraction can't find anything to cut. It only
deletes tokens (never rewrites), so there is no hallucination risk.

The model is loaded lazily on first use and downloads to the HuggingFace cache. Install the
backend with ``pip install llmlingua`` (or ``tokengate[compression]``).
"""

from __future__ import annotations

from tokengate.budgeting.token_counter import TokenCounter, resolve_counter
from tokengate.core.block import TokenBlock
from tokengate.models.base import resolve_device
from tokengate.utils.logging import get_logger, log_event

_logger = get_logger("compression.llmlingua")

DEFAULT_LLMLINGUA_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

# Structural tokens LLMLingua-2 must never drop (keeps sentences/numbers readable).
_FORCE_TOKENS = ["\n", ".", "!", "?", ",", ":", ";"]


class LLMLinguaCompressor:
    """:class:`~tokengate.compression.base.Compressor` backed by LLMLingua-2."""

    def __init__(
        self,
        model_name: str = DEFAULT_LLMLINGUA_MODEL,
        *,
        counter: TokenCounter | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._counter = counter
        self.device = device or resolve_device()
        self._compressor: object | None = None

    @property
    def compressor(self) -> object:
        """Lazily construct the LLMLingua PromptCompressor (downloads weights on first use)."""
        if self._compressor is None:
            from llmlingua import PromptCompressor

            log_event(_logger, "llmlingua_loading", model=self.model_name, device=self.device)
            self._compressor = PromptCompressor(
                model_name=self.model_name,
                use_llmlingua2=True,
                device_map=self.device,
            )
            log_event(_logger, "llmlingua_loaded", model=self.model_name)
        return self._compressor

    def compress_block(
        self, block: TokenBlock, query: str, *, keep_ratio: float
    ) -> TokenBlock:
        if not block.compressible:
            return block
        counter = resolve_counter(self._counter)
        original_tokens = block.ensure_token_count(counter)

        # LLMLingua's `rate` is the fraction of tokens to KEEP. force_reserve_digit protects
        # numbers (amounts, dates) — exactly the content relevance scoring throws away.
        rate = min(max(keep_ratio, 0.05), 1.0)
        try:
            result = self.compressor.compress_prompt(  # type: ignore[attr-defined]
                block.content,
                rate=rate,
                force_tokens=_FORCE_TOKENS,
                force_reserve_digit=True,
            )
        except Exception as exc:  # compression must never break the pipeline
            log_event(_logger, "llmlingua_compress_failed", error=str(exc))
            return block

        new_content = (
            result.get("compressed_prompt", "") if isinstance(result, dict) else str(result)
        ).strip()
        if not new_content or new_content == block.content:
            return block
        new_tokens = counter.count(new_content)
        if new_tokens >= original_tokens:  # no real gain — keep the original
            return block

        metadata = dict(block.metadata)
        metadata["compressed"] = True
        metadata["original_token_count"] = original_tokens
        metadata["compression_method"] = "llmlingua2"
        return block.copy(content=new_content, token_count=new_tokens, metadata=metadata)


__all__ = ["LLMLinguaCompressor", "DEFAULT_LLMLINGUA_MODEL"]
