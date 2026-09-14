"""Publish this project's training inputs to private Kaggle Datasets.

Pretraining runs on Kaggle Notebooks, which cannot read from this machine or
from a private GitHub repository. The inputs therefore have to be staged as
Kaggle Datasets and attached to the notebook.

Three datasets are created, deliberately kept separate:

``lma-hindi-packed``   Model H's packed token streams and its tokenizer.
``lma-nepali-packed``  Model L's packed token streams and its tokenizer.
``lma-code``           Both languages' code trees and configs.

The two corpora are kept in *separate* datasets rather than one combined
dataset, so the course requirement that the two models share no data, tokenizer
or vocabulary holds at the storage layer too: a notebook training Model H
attaches only Hindi's dataset and is physically unable to read Nepali's.

Note that Kaggle is transport, not a deliverable. The assignment requires
Google Drive links for large artifacts in the README, and requires that every
figure, log and table live in the git branch itself.

Usage
-----
    python scripts/kaggle_push.py --what code
    python scripts/kaggle_push.py --what hindi nepali
    python scripts/kaggle_push.py --what all --message "repacked with EOS"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directory names that hold corpus data rather than code. Anything under one of
# these is excluded from the code archive, so a stray .py in a data directory
# can never drag a multi-gigabyte sibling along with it.
CORPUS_DIRS = {
    "raw", "interim", "cleaned", "processed", "splits", "packed", "final",
    "metadata", "external",
}

# Kaggle dataset slugs must be lowercase, hyphenated, and 6-50 characters.
DATASETS: dict[str, dict] = {
    "hindi": {
        "slug": "lma-hindi-packed",
        "title": "LMA Model H (Hindi) packed corpus and tokenizer",
        "files": [
            "hindi/data/packed/train.bin",
            "hindi/data/packed/train.json",
            "hindi/data/packed/val.bin",
            "hindi/data/packed/val.json",
            "hindi/data/packed/test.bin",
            "hindi/data/packed/test.json",
            "hindi/tokenizer/hindi_unigram_8000.model",
            "hindi/tokenizer/hindi_unigram_8000.vocab",
        ],
    },
    "nepali": {
        "slug": "lma-nepali-packed",
        "title": "LMA Model L (Nepali) packed corpus and tokenizer",
        "files": [
            "nepali/data/packed/train.bin",
            "nepali/data/packed/train.json",
            "nepali/data/packed/val.bin",
            "nepali/data/packed/val.json",
            "nepali/data/packed/test.bin",
            "nepali/data/packed/test.json",
            "nepali/tokenizer/nepali_bpe.model",
            "nepali/tokenizer/nepali_bpe.vocab",
        ],
    },
    "code": {
        "slug": "lma-code",
        "title": "LMA project code: model, training, evaluation",
        # Packed into a single archive rather than uploaded as loose
        # directories. Kaggle's CLI compresses subdirectories on upload and
        # whether it re-extracts them is not something to leave to chance when
        # `import hindi.model.transformer` has to work; building the archive
        # here makes the notebook's unzip step deterministic.
        #
        # Both languages' code ships in one archive purely so a single dataset
        # can back either notebook run. They remain separate package trees and
        # neither imports the other; the corpora, which are what the
        # independence requirement is actually about, stay in separate datasets.
        "archive": "lma_code.zip",
        # Explicit, non-recursive globs. A "**" pattern here would walk into
        # hindi/data/splits/, which holds ~676,000 individual .txt files, and
        # take minutes to enumerate for no benefit.
        "archive_globs": [
            "hindi/*.py", "hindi/model/*.py", "hindi/train/*.py",
            "hindi/eval/*.py", "hindi/tests/*.py", "hindi/data/*.py",
            "nepali/*.py", "nepali/model/*.py", "nepali/train/*.py",
            "nepali/eval/*.py", "nepali/tests/*.py", "nepali/data/*.py",
            "hindi/configs/*.yaml", "nepali/configs/*.yaml",
            # Phase 3. The reasoning corpora are small enough (a few MB of
            # JSONL) to travel with the code, which is what lets a finetuning
            # kernel run without mounting the multi-hundred-megabyte pretraining
            # corpus it does not need.
            "hindi/reasoning/*.py", "nepali/reasoning/*.py",
            "hindi/reasoning/data/*.jsonl", "nepali/reasoning/data/*.jsonl",
            "hindi/reasoning/data/*.json", "nepali/reasoning/data/*.json",
            # Finetuning must reuse each language's Phase 1 tokenizer, and the
            # only other copy lives in that language's packed-corpus dataset.
            # Shipping the models here (~500 KB each) keeps that mount optional.
            "hindi/tokenizer/*.model", "nepali/tokenizer/*.model",
        ],
        "archive_files": ["requirements.txt"],
    },
}


def kaggle_cli() -> list[str]:
    """Return the command prefix for invoking the Kaggle CLI.

    Prefers the interpreter's own ``kaggle`` module so the call uses the same
    virtual environment as the rest of the project.

    Returns:
        Argument list to prefix Kaggle subcommands with.
    """
    return [sys.executable, "-m", "kaggle"]


def kaggle_username() -> str:
    """Look up the authenticated Kaggle username.

    Returns:
        The username string.

    Raises:
        RuntimeError: If the CLI is not authenticated.
    """
    result = subprocess.run(
        kaggle_cli() + ["config", "view"],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if "username:" in line:
            name = line.split("username:", 1)[1].strip()
            if name and name != "None":
                return name
    raise RuntimeError(
        "Kaggle CLI is not authenticated. Write your token to "
        "~/.kaggle/access_token or set KAGGLE_API_TOKEN, then retry."
    )


def stage(name: str, spec: dict, staging: Path, username: str) -> Path:
    """Assemble one dataset's payload and metadata in a staging directory.

    Files are hard-linked rather than copied where the filesystem allows it,
    because the packed token streams are ~1GB each and duplicating them just to
    upload them is wasteful.

    Args:
        name: Short dataset key (``"hindi"``, ``"nepali"``, ``"code"``).
        spec: The entry from :data:`DATASETS`.
        staging: Directory to build the payload in.
        username: Authenticated Kaggle username, used to build the dataset id.

    Returns:
        The staging directory that was populated.

    Raises:
        FileNotFoundError: If a declared input file is missing.
    """
    staging.mkdir(parents=True, exist_ok=True)

    for rel in spec.get("files", []):
        source = REPO_ROOT / rel
        if not source.exists():
            raise FileNotFoundError(
                f"{name}: missing {rel}. For packed streams, run "
                f"`python scripts/pack_corpus.py --lang {name}` first."
            )
        # Flatten into the dataset root, but keep the language prefix so the two
        # corpora can never be confused once mounted on Kaggle.
        dest = staging / Path(rel).name
        if dest.exists():
            dest.unlink()
        try:
            os.link(source, dest)
        except OSError:
            shutil.copy2(source, dest)

    if spec.get("archive"):
        # Paths inside the archive are kept relative to the repository root, so
        # unzipping it on Kaggle reproduces the repo layout and
        # `import hindi.model.transformer` works from the extraction directory.
        import zipfile

        archive_path = staging / spec["archive"]
        written = 0
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for pattern in spec.get("archive_globs", []):
                for path in sorted(REPO_ROOT.glob(pattern)):
                    parts = path.relative_to(REPO_ROOT).parts
                    # Never sweep corpus data into the code archive.
                    if "__pycache__" in parts or set(parts) & CORPUS_DIRS:
                        continue
                    zf.write(path, path.relative_to(REPO_ROOT).as_posix())
                    written += 1
            for rel in spec.get("archive_files", []):
                source = REPO_ROOT / rel
                if not source.exists():
                    raise FileNotFoundError(f"{name}: missing {rel}")
                zf.write(source, rel)
                written += 1
        print(f"  built {spec['archive']} from {written} files "
              f"({archive_path.stat().st_size/1e3:.0f} KB)")

    metadata = {
        "title": spec["title"],
        "id": f"{username}/{spec['slug']}",
        "licenses": [{"name": "other"}],
    }
    (staging / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return staging


def dataset_exists(dataset_id: str) -> bool:
    """Check whether a dataset already exists under the account.

    Args:
        dataset_id: Full ``username/slug`` identifier.

    Returns:
        True if the dataset is listed, else False.
    """
    result = subprocess.run(
        kaggle_cli() + ["datasets", "list", "--mine", "-s", dataset_id.split("/")[-1]],
        capture_output=True,
        text=True,
    )
    return dataset_id in result.stdout


def push(name: str, message: str, dry_run: bool = False) -> None:
    """Create or version one dataset on Kaggle.

    Args:
        name: Short dataset key.
        message: Version message, recorded in the dataset's history.
        dry_run: If True, stage and print the command without uploading.
    """
    spec = DATASETS[name]
    username = kaggle_username()
    dataset_id = f"{username}/{spec['slug']}"

    with tempfile.TemporaryDirectory(prefix=f"kaggle_{name}_") as tmp:
        staging = stage(name, spec, Path(tmp) / "payload", username)
        payload_bytes = sum(
            f.stat().st_size for f in staging.rglob("*") if f.is_file()
        )
        print(f"\n[{name}] {dataset_id}")
        print(f"  staged {payload_bytes / 1e9:.2f} GB in {staging}")

        if dataset_exists(dataset_id):
            cmd = kaggle_cli() + [
                "datasets", "version", "-p", str(staging), "-m", message,
                "--dir-mode", "skip",
            ]
            print(f"  dataset exists -> creating a new version")
        else:
            cmd = kaggle_cli() + [
                "datasets", "create", "-p", str(staging), "--dir-mode", "skip",
            ]
            print(f"  dataset is new -> creating (private by default)")

        if dry_run:
            print(f"  DRY RUN: {' '.join(cmd)}")
            return

        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            print(f"  FAILED with exit code {result.returncode}")
        else:
            print(f"  done -> https://www.kaggle.com/datasets/{dataset_id}")


def main() -> None:
    """Parse arguments and push the requested datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--what",
        nargs="+",
        default=["all"],
        choices=["all", "hindi", "nepali", "code"],
        help="Which datasets to push.",
    )
    parser.add_argument(
        "--message", default="update", help="Version message for existing datasets."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Stage but do not upload."
    )
    args = parser.parse_args()

    targets = ["hindi", "nepali", "code"] if "all" in args.what else args.what
    for name in targets:
        push(name, args.message, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
