"""
Downloaded-lane collectors: Sangraha and IndicCorp v2.

These are the sources where a third party performed selection, extraction and
cleaning. Per the course clarification they are *not* manual, and per the lane
invariant they may not overlap the manual lane's content families -- which is
why Wikipedia is collected only through the API in the manual lane and the
HuggingFace ``wikimedia/wikipedia`` dump is not collected at all.

Both collectors stream. Sangraha's Hindi ``verified`` split alone is 100
parquet shards; materialising it would consume the disk budget for no benefit
when the pipeline reads documents one at a time anyway.

A note on residual overlap
--------------------------
Sangraha and IndicCorp are built from web crawls that include news sites and
Wikipedia. Some of their documents will therefore resemble documents in the
manual lane. That overlap cannot be eliminated by source selection -- only
measured. The dedup stage resolves cross-lane collisions *in favour of the
manual copy* and reports the rate, which is both the honest treatment and a
substantive methodology result for the report.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from .. import schema
from ..textnorm import devanagari_ratio, normalize
from .base import Budget, Collector

# Streaming datasets emit a lot of progress chatter that would drown the
# overnight console; the driver sets these before importing `datasets`.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class SangrahaCollector(Collector):
    """Stream one Sangraha subset for one language.

    ``spec.params``:

    ``subset``
        ``verified`` (web + OCR + transcribed, human-checked) or ``unverified``
        (perplexity-filtered from other corpora). ``synthetic`` is never
        collected -- it is machine-translated text, and translationese in a
        pretraining corpus teaches the model a register no human writes.
    ``lang_dir``
        Sangraha's own language code: ``hin`` for Hindi, ``nep`` for Nepali.
    """

    REPO = "ai4bharat/sangraha"

    def preflight(self) -> str | None:
        subset = self.spec.params.get("subset")
        if subset == "synthetic":
            return "synthetic subset is machine-translated and excluded by policy"
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return "huggingface_hub not installed"
        try:
            files = HfApi().list_repo_files(self.REPO, repo_type="dataset")
        except Exception as exc:  # noqa: BLE001
            return f"cannot list {self.REPO}: {type(exc).__name__}: {exc}"
        prefix = f"{subset}/{self.spec.params['lang_dir']}/"
        if not any(f.startswith(prefix) for f in files):
            return f"{prefix} not present in {self.REPO}"
        return None

    def iter_documents(self, budget: Budget) -> Iterator[schema.Document]:
        from datasets import load_dataset

        subset = self.spec.params["subset"]
        lang_dir = self.spec.params["lang_dir"]
        min_chars = int(self.spec.params.get("min_chars", 250))
        min_deva = float(self.spec.params.get("min_devanagari_ratio", 0.55))

        stream = load_dataset(
            self.REPO,
            data_dir=f"{subset}/{lang_dir}",
            split="train",
            streaming=True,
        )

        for index, row in enumerate(stream):
            if budget.remaining_seconds <= 0:
                break
            raw = row.get("text") or row.get("content") or ""
            if not raw:
                continue
            text = normalize(raw)
            if len(text) < min_chars or devanagari_ratio(text) < min_deva:
                continue
            yield self.make_document(
                text,
                title=None,
                origin_url=row.get("url"),
                meta={
                    "hf_repo": self.REPO,
                    "subset": subset,
                    "lang_dir": lang_dir,
                    "row_index": index,
                    "upstream_doc_id": row.get("doc_id") or row.get("id"),
                },
            )


class IndicCorpCollector(Collector):
    """Stream IndicCorp v2 plain-text shards for one language.

    IndicCorp ships one sentence per line rather than one document per line, so
    lines are reassembled into paragraph-sized pseudo-documents. Without this
    the corpus would be millions of single-sentence "documents", which breaks
    document-level deduplication and produces meaningless split statistics.
    """

    REPO = "ai4bharat/IndicCorpV2"

    def preflight(self) -> str | None:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return "huggingface_hub not installed"
        try:
            files = HfApi().list_repo_files(self.REPO, repo_type="dataset")
        except Exception as exc:  # noqa: BLE001
            return f"cannot list {self.REPO}: {type(exc).__name__}: {exc}"
        wanted = set(self.spec.params.get("files", []))
        missing = wanted - set(files)
        if missing:
            return f"missing from {self.REPO}: {sorted(missing)}"
        return None

    def iter_documents(self, budget: Budget) -> Iterator[schema.Document]:
        from huggingface_hub import hf_hub_download

        files = self.spec.params.get("files", [])
        lines_per_doc = int(self.spec.params.get("lines_per_doc", 12))
        min_chars = int(self.spec.params.get("min_chars", 250))
        min_deva = float(self.spec.params.get("min_devanagari_ratio", 0.55))

        for file_name in files:
            if budget.remaining_seconds <= 0:
                break
            self.report(status="RUNNING", phase=f"downloading {file_name}")
            local = hf_hub_download(
                self.REPO, file_name, repo_type="dataset", resume_download=True
            )
            self.report(status="RUNNING", phase=f"reading {file_name}")

            buffer: list[str] = []
            with open(local, "r", encoding="utf-8", errors="replace") as handle:
                for line_no, line in enumerate(handle):
                    if line_no % 20_000 == 0 and budget.remaining_seconds <= 0:
                        break
                    line = line.strip()
                    # A blank line is IndicCorp's document boundary; honour it
                    # when present and otherwise chunk by line count.
                    if not line:
                        if buffer:
                            doc = self._emit(buffer, file_name, min_chars, min_deva)
                            buffer = []
                            if doc is not None:
                                yield doc
                        continue
                    buffer.append(line)
                    if len(buffer) >= lines_per_doc:
                        doc = self._emit(buffer, file_name, min_chars, min_deva)
                        buffer = []
                        if doc is not None:
                            yield doc
            if buffer:
                doc = self._emit(buffer, file_name, min_chars, min_deva)
                if doc is not None:
                    yield doc

    def _emit(
        self, buffer: list[str], file_name: str, min_chars: int, min_deva: float
    ) -> schema.Document | None:
        text = normalize("\n".join(buffer))
        if len(text) < min_chars or devanagari_ratio(text) < min_deva:
            return None
        return self.make_document(
            text,
            meta={"hf_repo": self.REPO, "file": file_name, "lines": len(buffer)},
        )
