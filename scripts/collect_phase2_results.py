"""Pull a finished Kaggle run's artifacts into report/phase2/ for committing.

The assignment is explicit that graders will not follow external links to find
figures: "if it is not in the branch, it is not graded". So everything small
enough to version -- loss curves, attention heatmaps, metric JSON, training
logs, the resolved run config -- is copied into ``report/phase2/`` here and
committed. Only the checkpoints stay external, and those need Google Drive
links in the README.

Usage
-----
    python scripts/collect_phase2_results.py --lang hindi
    python scripts/collect_phase2_results.py --lang hindi nepali --summary
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "report/phase2"
FIG_DIR = REPORT_DIR / "figures"

KERNELS = {
    "hindi": "lma-phase-2-model-h-hindi",
    "nepali": "lma-phase-2-model-l-nepali",
}


def kaggle_cli() -> list[str]:
    """Return the command prefix for invoking the Kaggle CLI."""
    return [sys.executable, "-m", "kaggle"]


def kaggle_username() -> str:
    """Look up the authenticated Kaggle username.

    Returns:
        The username string.

    Raises:
        RuntimeError: If the CLI is not authenticated.
    """
    out = subprocess.run(kaggle_cli() + ["config", "view"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "username:" in line:
            name = line.split("username:", 1)[1].strip()
            if name and name != "None":
                return name
    raise RuntimeError("Kaggle CLI is not authenticated.")


def download_output(lang: str, dest: Path) -> Path:
    """Download one kernel's output files.

    Args:
        lang: Language whose kernel to fetch.
        dest: Directory to download into.

    Returns:
        The directory the files were written to.
    """
    kernel_id = f"{kaggle_username()}/{KERNELS[lang]}"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        kaggle_cli() + ["kernels", "output", kernel_id, "-p", str(dest)],
        capture_output=True, text=True,
    )
    return dest


def harvest(lang: str, downloaded: Path) -> list[Path]:
    """Copy a run's small artifacts into report/phase2/.

    Handles both shapes the kernel can produce: the bundled
    ``report_phase2.zip`` if it was written, and loose files otherwise (which is
    what a run cut short before its final cell leaves behind).

    Args:
        lang: Language the artifacts belong to.
        downloaded: Directory holding the downloaded kernel output.

    Returns:
        The list of paths written into the report directory.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    bundle = downloaded / "report_phase2.zip"
    staging = downloaded
    if bundle.is_file() and zipfile.is_zipfile(bundle):
        staging = downloaded / "_unzipped"
        staging.mkdir(exist_ok=True)
        with zipfile.ZipFile(bundle) as zf:
            zf.extractall(staging)

    for png in sorted(staging.rglob("*.png")):
        target = FIG_DIR / png.name
        shutil.copy2(png, target)
        written.append(target)

    for pattern, rename in (
        (f"{lang}_eval.json", f"{lang}_eval.json"),
        (f"{lang}_attention.json", f"{lang}_attention.json"),
        ("training_log.jsonl", f"{lang}_training_log.jsonl"),
        (f"{lang}_training_log.jsonl", f"{lang}_training_log.jsonl"),
        ("config.yaml", f"{lang}_config.yaml"),
        (f"{lang}_config.yaml", f"{lang}_config.yaml"),
    ):
        for found in sorted(staging.rglob(pattern)):
            target = REPORT_DIR / rename
            shutil.copy2(found, target)
            written.append(target)
            break

    # The kernel log itself is the training record: keep it, since the spec
    # asks for training logs in the branch.
    for log in sorted(downloaded.glob("*.log")):
        target = REPORT_DIR / f"{lang}_kernel.log.json"
        shutil.copy2(log, target)
        written.append(target)

    return written


def summarise() -> str:
    """Build a side-by-side metrics table from whatever eval JSON is present.

    Returns:
        A Markdown table, or a note if no results have landed yet.
    """
    reports = {}
    for lang in KERNELS:
        path = REPORT_DIR / f"{lang}_eval.json"
        if path.is_file():
            reports[lang] = json.loads(path.read_text(encoding="utf-8"))
    if not reports:
        return "_No evaluation results collected yet._"

    langs = list(reports)
    rows = [
        ("Parameters (total)",
         lambda r: f"{r['parameter_count']['total']:,}"),
        ("Parameters (non-embedding)",
         lambda r: f"{r['parameter_count']['non_embedding']:,}"),
        ("Vocabulary", lambda r: f"{r['model_config']['vocab_size']:,}"),
        ("Checkpoint step", lambda r: f"{r['checkpoint_step']:,}"),
        ("Test cross-entropy (nats)",
         lambda r: f"{r['intrinsic']['cross_entropy_nats']:.4f}"),
        ("Test perplexity", lambda r: f"{r['intrinsic']['perplexity']:.2f}"),
        ("**Bits per byte**", lambda r: f"**{r['intrinsic']['bits_per_byte']:.4f}**"),
        ("Tokens scored", lambda r: f"{r['intrinsic']['tokens_scored']:,}"),
    ]
    header = "| Metric | " + " | ".join(
        ("Model H (Hindi)" if l == "hindi" else "Model L (Nepali)") for l in langs
    ) + " |"
    sep = "|---|" + "---:|" * len(langs)
    lines = [header, sep]
    for label, fn in rows:
        lines.append("| " + label + " | " +
                     " | ".join(fn(reports[l]) for l in langs) + " |")

    lines.append("")
    lines.append("### Generation metrics")
    lines.append("| Model | Decoding | BLEU | chrF++ | ROUGE-L | Repetition | Distinct-1 | Distinct-2 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for lang in langs:
        for name, m in reports[lang]["generation"]["settings"].items():
            lines.append(
                f"| {lang} | {name} | {m['bleu']:.2f} | {m['chrf_pp']:.2f} | "
                f"{m['rouge_l']:.3f} | {m['repetition_rate']:.3f} | "
                f"{m['distinct_1']:.3f} | {m['distinct_2']:.3f} |"
            )

    lines.append("")
    lines.append("### Attention")
    lines.append("| Model | Mean entropy | Uniform baseline | Mean distance | Local-head fraction |")
    lines.append("|---|---:|---:|---:|---:|")
    for lang in langs:
        a = reports[lang].get("attention")
        if not a:
            continue
        lines.append(
            f"| {lang} | {a['mean_entropy']:.3f} | "
            f"{a.get('uniform_baseline_entropy', float('nan')):.3f} | "
            f"{a['mean_distance']:.2f} | {a['local_head_fraction']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Fetch, harvest and summarise the requested languages' runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", nargs="+", default=["hindi", "nepali"],
                        choices=sorted(KERNELS))
    parser.add_argument("--summary", action="store_true",
                        help="Print the side-by-side metrics table.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Summarise what is already collected.")
    args = parser.parse_args()

    if not args.skip_download:
        for lang in args.lang:
            with tempfile.TemporaryDirectory(prefix=f"collect_{lang}_") as tmp:
                out = download_output(lang, Path(tmp))
                written = harvest(lang, out)
                print(f"[{lang}] collected {len(written)} artifact(s):")
                for w in written:
                    print(f"    {w.relative_to(REPO_ROOT)}")

    if args.summary or args.skip_download:
        print()
        print(summarise())


if __name__ == "__main__":
    main()
