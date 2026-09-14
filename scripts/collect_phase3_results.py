"""Pull finished Phase 3 Kaggle runs into report/phase3/ for committing.

The assignment is explicit that graders will not follow external links to find
figures: "if it is not in the branch, it is not graded". So everything small
enough to version -- the reasoning metric JSONs, the attention comparison,
finetuning logs, the resolved config, and the heatmaps -- is copied into
``report/phase3/`` here and committed. Only the finetuned checkpoints stay
external, and those need Google Drive links in the README.

``--summary`` prints the pretrained-vs-finetuned comparison table for both
models side by side, which is the table the final report is built around.

Usage
-----
    python scripts/collect_phase3_results.py --lang hindi nepali
    python scripts/collect_phase3_results.py --summary
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Every path this script prints can contain Devanagari, and a Windows console
# defaults to cp1252, which cannot encode it. Without this the whole collection
# succeeds and then dies on the final print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "report/phase3"
FIG_DIR = REPORT_DIR / "figures"

KERNELS = {
    "hindi": "lma-phase-3-finetune-model-h-hindi",
    "nepali": "lma-phase-3-finetune-model-l-nepali",
}

MODELS = {"hindi": "Model H", "nepali": "Model L"}


def kaggle_cli() -> list[str]:
    """Return the command prefix for invoking the Kaggle CLI.

    Returns:
        Argv prefix that runs the Kaggle CLI in this interpreter.
    """
    return [sys.executable, "-m", "kaggle"]


def kaggle_username() -> str:
    """Look up the authenticated Kaggle username.

    Asks the CLI rather than reading ``kaggle.json`` directly, so this agrees
    with whatever credentials the CLI is actually using -- environment
    variables, a config directory override, or the default file.

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
    """Download one language's finetuning kernel output.

    Args:
        lang: Target language.
        dest: Directory to download into.

    Returns:
        The directory the files landed in.

    Raises:
        RuntimeError: If the download fails.
    """
    import os

    kernel_id = f"{kaggle_username()}/{KERNELS[lang]}"
    dest.mkdir(parents=True, exist_ok=True)
    # The Kaggle CLI prints each downloaded path, and the kernel output contains
    # Devanagari filenames. On a Windows console the child process inherits
    # cp1252, cannot encode them, and dies *after* downloading everything -- so
    # the failure looks like a download error when the files are already there.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        kaggle_cli() + ["kernels", "output", kernel_id, "-p", str(dest)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env,
    )
    output = (result.stdout + result.stderr).strip()
    downloaded = sorted(dest.rglob("*"))
    print(f"  {len(downloaded)} paths under {dest}")
    # The CLI can still die while *printing* its file list even with UTF-8
    # forced, since some of its output goes through a stream this process does
    # not control. Judge success by whether the files arrived, not by the exit
    # code -- otherwise a completed download is reported as a failure.
    if not downloaded:
        raise RuntimeError(f"could not download {kernel_id}: {output}")
    if result.returncode != 0:
        print(f"  note: CLI exited {result.returncode} but files are present")
    return dest


def harvest(lang: str, downloaded: Path) -> list[Path]:
    """Copy the committable artifacts out of a downloaded kernel output.

    The kernel writes a ``report_phase3_<lang>.zip`` holding the JSON reports,
    figures and logs. Preferring that archive over loose files keeps this in
    step with whatever the notebook decided to bundle.

    Args:
        lang: Target language.
        downloaded: Directory holding the kernel output.

    Returns:
        The paths written into ``report/phase3/``.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    bundles = sorted(downloaded.rglob(f"report_phase3_{lang}.zip"))
    if bundles:
        with zipfile.ZipFile(bundles[0]) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                target = (FIG_DIR / Path(name).name if name.startswith("figures/")
                          else REPORT_DIR / Path(name).name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                written.append(target)
    else:
        # Fall back to loose files if the bundle is missing, e.g. because the
        # kernel stopped after evaluation but before the bundling cell.
        for pattern in (f"{lang}_reasoning_*.json",
                        f"{lang}_attention_comparison.json",
                        f"{lang}_finetune_log.jsonl",
                        f"{lang}_finetune_config.yaml"):
            for found in downloaded.rglob(pattern):
                target = REPORT_DIR / found.name
                shutil.copy2(found, target)
                written.append(target)
        for found in downloaded.rglob(f"{lang}_*.png"):
            target = FIG_DIR / found.name
            shutil.copy2(found, target)
            written.append(target)

    for path in written:
        print(f"    {path.relative_to(REPO_ROOT)}")
    return written


def load(lang: str, tag: str) -> dict | None:
    """Read one reasoning report if it exists.

    Args:
        lang: Language name.
        tag: ``"pretrained"`` or ``"finetuned"``.

    Returns:
        The parsed report, or ``None``.
    """
    path = REPORT_DIR / f"{lang}_reasoning_{tag}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarise() -> str:
    """Build the side-by-side pretrained-vs-finetuned table.

    Returns:
        The table as Markdown.
    """
    rows: list[str] = []
    rows.append("| Model | Metric | Pretrained | Finetuned | Change |")
    rows.append("|---|---|---:|---:|---:|")
    for lang in ("hindi", "nepali"):
        pre, post = load(lang, "pretrained"), load(lang, "finetuned")
        if not pre or not post:
            rows.append(f"| {MODELS[lang]} | _results not collected yet_ "
                        "| | | |")
            continue
        name = f"{MODELS[lang]} ({lang.title()})"
        for label, path in (
            ("Exact match", ("exact_match", "accuracy")),
            ("Forced choice", ("forced_choice", "accuracy")),
        ):
            a = pre[path[0]][path[1]]
            b = post[path[0]][path[1]]
            rows.append(f"| {name} | {label} | {a:.2%} | {b:.2%} | "
                        f"{b - a:+.2%} |")
            name = ""
        a = pre["answer_token_loss"]["answer_token_loss"]
        b = post["answer_token_loss"]["answer_token_loss"]
        rows.append(f"| | Answer-token loss | {a:.4f} | {b:.4f} | "
                    f"{b - a:+.4f} |")
        chance = post["forced_choice"]["chance_accuracy"]
        rows.append(f"| | _forced-choice chance_ | {chance:.2%} | "
                    f"{chance:.2%} | — |")

    table = "\n".join(rows)

    detail: list[str] = ["", "Per task family (exact match):", "",
                         "| Model | Task | Hops | Pretrained | Finetuned |",
                         "|---|---|---:|---:|---:|"]
    for lang in ("hindi", "nepali"):
        pre, post = load(lang, "pretrained"), load(lang, "finetuned")
        if not pre or not post:
            continue
        name = MODELS[lang]
        for task in sorted(post["exact_match"]["by_task"]):
            hops = 2 if task == "transitive3" else 1
            a = pre["exact_match"]["by_task"][task]["accuracy"]
            b = post["exact_match"]["by_task"][task]["accuracy"]
            detail.append(f"| {name} | `{task}` | {hops} | {a:.2%} | {b:.2%} |")
            name = ""
    return table + "\n" + "\n".join(detail)


def main() -> None:
    """Download, harvest and optionally summarise the Phase 3 runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", nargs="*", default=[], choices=sorted(KERNELS))
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--keep", default=None,
                        help="keep the raw download in this directory")
    args = parser.parse_args()

    for lang in args.lang:
        print(f"[{lang}] downloading {KERNELS[lang]}")
        if args.keep:
            dest = Path(args.keep) / lang
            download_output(lang, dest)
            harvest(lang, dest)
        else:
            with tempfile.TemporaryDirectory(prefix=f"phase3_{lang}_") as tmp:
                dest = Path(tmp)
                download_output(lang, dest)
                harvest(lang, dest)

    if args.summary or not args.lang:
        print()
        print(summarise())


if __name__ == "__main__":
    main()
