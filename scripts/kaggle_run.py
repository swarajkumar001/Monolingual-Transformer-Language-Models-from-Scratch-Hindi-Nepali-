"""Launch and monitor one language's pretraining run as a Kaggle kernel.

Kaggle notebooks can be pushed and executed through the API rather than driven
by hand in the browser, which is what this does: it takes
``notebooks/kaggle_pretrain.ipynb``, substitutes the target language, attaches
exactly that language's dataset, and starts a GPU run.

One kernel per language, with separate slugs and separate attached datasets.
Model H's kernel has no access to Model L's corpus and vice versa — the
independence requirement is enforced by what is mounted, not by convention.

Usage
-----
    python scripts/kaggle_run.py --lang hindi            # push and start
    python scripts/kaggle_run.py --lang hindi --status   # check progress
    python scripts/kaggle_run.py --lang hindi --fetch    # download output
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks/kaggle_pretrain.ipynb"

# Kaggle derives a kernel's slug from its TITLE and silently ignores the slug in
# the id field, so the two must be kept consistent by hand or the kernel ends up
# at an address the API cannot then poll. slugify() below asserts they agree.
LANGUAGES = {
    "hindi": {"slug": "lma-phase-2-model-h-hindi",
              "title": "LMA Phase 2 Model H Hindi",
              "dataset": "lma-hindi-packed"},
    "nepali": {"slug": "lma-phase-2-model-l-nepali",
               "title": "LMA Phase 2 Model L Nepali",
               "dataset": "lma-nepali-packed"},
}


def slugify(title: str) -> str:
    """Reproduce Kaggle's title-to-slug rule.

    Args:
        title: The kernel title.

    Returns:
        The slug Kaggle will actually use.
    """
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")


for _lang, _spec in LANGUAGES.items():
    assert slugify(_spec["title"]) == _spec["slug"], (
        f"{_lang}: title {_spec['title']!r} slugifies to "
        f"{slugify(_spec['title'])!r}, not {_spec['slug']!r}; the kernel would be "
        "created at an address --status and --fetch cannot reach"
    )


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


def build_notebook(lang: str) -> dict:
    """Load the notebook and pin it to one language.

    The notebook carries ``LANGUAGE = "hindi"`` as its editable default. For a
    batch run there is nobody to edit it, so the constant is rewritten here.

    Args:
        lang: Target language.

    Returns:
        The notebook as a dict, with the language constant substituted.

    Raises:
        RuntimeError: If the language constant could not be found.
    """
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    patched = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for i, line in enumerate(cell["source"]):
            if re.match(r'^LANGUAGE\s*=\s*"', line):
                cell["source"][i] = re.sub(
                    r'^LANGUAGE\s*=\s*"[a-z]+"',
                    f'LANGUAGE = "{lang}"',
                    line,
                )
                patched += 1
    if patched != 1:
        raise RuntimeError(
            f"expected exactly one LANGUAGE assignment, patched {patched}"
        )
    validate_notebook(nb)
    return nb


def validate_notebook(nb: dict) -> None:
    """Parse every code cell exactly as Jupyter will join it.

    nbformat stores a cell's source as a list of lines, and every element
    except the last must carry its own trailing newline. A generator that
    strips them produces a notebook that looks fine in JSON but collapses into
    one unbroken line when executed. That failure cost a GPU session once; this
    check runs before every push so it cannot happen silently again.

    Args:
        nb: The notebook as a dict.

    Raises:
        SyntaxError: If any code cell fails to parse as joined.
    """
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])  # exactly how Jupyter joins it
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise SyntaxError(
                f"code cell {i} does not parse as Jupyter would join it: {exc}"
            ) from exc


def push(lang: str, dry_run: bool = False) -> str:
    """Push the kernel for one language and start it running.

    Args:
        lang: Target language.
        dry_run: Stage the kernel directory but do not upload.

    Returns:
        The kernel's full ``username/slug`` id.
    """
    spec = LANGUAGES[lang]
    user = kaggle_username()
    kernel_id = f"{user}/{spec['slug']}"

    with tempfile.TemporaryDirectory(prefix=f"kernel_{lang}_") as tmp:
        stage = Path(tmp)
        nb = build_notebook(lang)
        (stage / "kaggle_pretrain.ipynb").write_text(
            json.dumps(nb, indent=1), encoding="utf-8"
        )
        metadata = {
            "id": kernel_id,
            "title": spec["title"],
            "code_file": "kaggle_pretrain.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            # Only this language's corpus is attached. The notebook additionally
            # raises if it ever sees the other one mounted.
            "dataset_sources": [f"{user}/lma-code", f"{user}/{spec['dataset']}"],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (stage / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        print(f"\n[{lang}] {kernel_id}")
        print(f"  datasets: {', '.join(metadata['dataset_sources'])}")
        if dry_run:
            print(f"  DRY RUN, staged in {stage}")
            print(json.dumps(metadata, indent=2))
            return kernel_id

        result = subprocess.run(
            kaggle_cli() + ["kernels", "push", "-p", str(stage)],
            capture_output=True, text=True,
        )
        print(result.stdout.strip() or result.stderr.strip())
        if result.returncode != 0:
            print(f"  push FAILED (exit {result.returncode})")
        else:
            print(f"  running -> https://www.kaggle.com/code/{kernel_id}")
    return kernel_id


def status(lang: str) -> str:
    """Report the current status of one language's kernel.

    Args:
        lang: Target language.

    Returns:
        The raw status string reported by Kaggle.
    """
    user = kaggle_username()
    kernel_id = f"{user}/{LANGUAGES[lang]['slug']}"
    result = subprocess.run(
        kaggle_cli() + ["kernels", "status", kernel_id],
        capture_output=True, text=True,
    )
    text = (result.stdout + result.stderr).strip()
    print(f"[{lang}] {text}")
    return text


def fetch(lang: str, out_dir: Path | None = None) -> None:
    """Download a finished kernel's output files.

    Args:
        lang: Target language.
        out_dir: Destination directory; defaults to ``kaggle_output/<lang>/``.
    """
    user = kaggle_username()
    kernel_id = f"{user}/{LANGUAGES[lang]['slug']}"
    dest = out_dir or (REPO_ROOT / "kaggle_output" / lang)
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        kaggle_cli() + ["kernels", "output", kernel_id, "-p", str(dest)],
        capture_output=True, text=True,
    )
    print(result.stdout.strip() or result.stderr.strip())
    print(f"[{lang}] output -> {dest}")


def main() -> None:
    """Parse arguments and push, poll, or fetch a language's kernel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True, choices=sorted(LANGUAGES))
    parser.add_argument("--status", action="store_true", help="Report status only.")
    parser.add_argument("--fetch", action="store_true", help="Download output only.")
    parser.add_argument("--dry-run", action="store_true", help="Stage but do not push.")
    args = parser.parse_args()

    if args.status:
        status(args.lang)
    elif args.fetch:
        fetch(args.lang)
    else:
        push(args.lang, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
