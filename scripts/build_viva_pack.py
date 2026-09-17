"""Bundle the whole project into a few text files for viva lookup.

The output is designed to be uploaded to a claude.ai Project, so that any chat
in that project can answer "where is X" and quote the exact line. Every source
line is numbered, and every file is preceded by its repository path, so an
answer can cite ``hindi/model/transformer.py:178`` and the line will match what
is open in the IDE.

Run::

    python scripts/build_viva_pack.py

Output lands in ``docs/viva_pack/`` (untracked).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "viva_pack"

# Each pack is (filename, human title, [repo-relative source paths]).
# Kept to six files because claude.ai Projects list them by name -- few, clearly
# named files are easier to point at than thirty.
PACKS: list[tuple[str, str, list[str]]] = [
    (
        "01_model_and_training.md",
        "Phase 2 - the Transformer and the training loop",
        [
            "hindi/model/transformer.py",
            "hindi/model/config.py",
            "hindi/train/train.py",
            "hindi/configs/model.yaml",
            "nepali/configs/model.yaml",
        ],
    ),
    (
        "02_evaluation_and_attention.md",
        "Phase 2 - evaluation, generation metrics, attention toolkit",
        [
            "hindi/eval/evaluate.py",
            "hindi/eval/attention.py",
            "hindi/eval/viz.py",
            "hindi/tests/test_model.py",
        ],
    ),
    (
        "03_data_and_tokenizer.md",
        "Phase 1 - cleaning, splitting, tokenizer",
        [
            "scripts/clean_lib.py",
            "hindi/data/pipeline/clean_hindi.py",
            "hindi/data/pipeline/split_hindi.py",
            "hindi/tokenizer/train_tokenizer.py",
            "hindi/tokenizer/analyze_tokenizer.py",
        ],
    ),
    (
        "04_nepali_pipeline.md",
        "Phase 1 - the Nepali stage pipeline (differs from Hindi)",
        [
            "nepali/configs/language.yaml",
            "nepali/scripts/s03_normalize.py",
            "nepali/scripts/s04_langid.py",
            "nepali/scripts/s07_holdout.py",
            "nepali/scripts/s08_dedup.py",
            "nepali/scripts/s09_decontaminate.py",
            "nepali/scripts/s11_finalize_splits.py",
            "nepali/scripts/s12_train_tokenizer.py",
        ],
    ),
    (
        "05_reasoning_finetuning.md",
        "Phase 3 - reasoning data, finetuning, evaluation",
        [
            "hindi/reasoning/generate.py",
            "hindi/reasoning/dataset.py",
            "hindi/reasoning/finetune.py",
            "hindi/reasoning/evaluate_reasoning.py",
            "hindi/reasoning/attention_compare.py",
            "hindi/configs/finetune.yaml",
        ],
    ),
    (
        "06_reports_and_numbers.md",
        "All reports and every reported number",
        [
            "README.md",
            "report/phase1/phase1_report.md",
            "report/phase2/phase2_report.md",
            "report/phase3/phase3_report.md",
            "report/phase3/README.md",
            "hindi/reasoning/data/dataset_stats.json",
            "nepali/reasoning/data/dataset_stats.json",
        ],
    ),
]

# Source files are fenced with a language hint so the code renders as code.
FENCE = {".py": "python", ".yaml": "yaml", ".yml": "yaml", ".json": "json"}


def render(rel: str) -> str:
    """Render one source file as a numbered, fenced section.

    Args:
        rel: Repository-relative path.

    Returns:
        A markdown section, or a short note if the file is missing.
    """
    path = REPO_ROOT / rel
    if not path.exists():
        return f"\n## {rel}\n\n*(not present in this repository)*\n"

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    # Markdown is included verbatim: numbering it would break its own tables and
    # fences, and the reports are read as prose rather than cited by line.
    if suffix in {".md"}:
        return f"\n## {rel}\n\n{text}\n"

    numbered = "\n".join(
        f"{i:4d}  {line}" for i, line in enumerate(text.splitlines(), start=1)
    )
    return f"\n## {rel}\n\n```{FENCE.get(suffix, '')}\n{numbered}\n```\n"


def main() -> None:
    """Write every pack file and print a summary table."""
    sys.stdout.reconfigure(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for name, title, sources in PACKS:
        parts = [
            f"# {title}",
            "",
            "Part of the LMA individual project viva pack. Line numbers below "
            "match the files in the repository exactly, so any line cited here "
            "can be opened directly in the IDE.",
            "",
            "Files in this pack:",
            "",
        ]
        parts += [f"- `{s}`" for s in sources]
        parts.append("")
        parts += [render(s) for s in sources]

        body = "\n".join(parts)
        (OUT_DIR / name).write_text(body, encoding="utf-8")
        size = len(body.encode("utf-8"))
        total += size
        print(f"  {name:<32} {size/1024:8.1f} KB  ({len(sources)} files)")

    print(f"\n  {'TOTAL':<32} {total/1024:8.1f} KB")
    print(f"\n  written to {OUT_DIR}")


if __name__ == "__main__":
    main()
