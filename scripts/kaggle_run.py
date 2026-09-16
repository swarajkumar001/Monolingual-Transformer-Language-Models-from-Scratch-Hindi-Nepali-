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
EVAL_NOTEBOOK = REPO_ROOT / "notebooks/kaggle_evaluate.ipynb"
FINETUNE_NOTEBOOK = REPO_ROOT / "notebooks/kaggle_finetune.ipynb"
ABLATION_NOTEBOOK = REPO_ROOT / "notebooks/kaggle_ablation.ipynb"

# Valid values include NvidiaTeslaT4, NvidiaTeslaP100 and Tpu1VmV38.
ACCELERATOR = "NvidiaTeslaT4"

# Kaggle derives a kernel's slug from its TITLE and silently ignores the slug in
# the id field, so the two must be kept consistent by hand or the kernel ends up
# at an address the API cannot then poll. slugify() below asserts they agree.
LANGUAGES = {
    "hindi": {"slug": "lma-phase-2-model-h-hindi",
              "title": "LMA Phase 2 Model H Hindi",
              "dataset": "lma-hindi-packed",
              "eval_slug": "lma-phase-2-eval-model-h-hindi",
              "eval_title": "LMA Phase 2 Eval Model H Hindi",
              "ft_slug": "lma-phase-3-finetune-model-h-hindi",
              "ft_title": "LMA Phase 3 Finetune Model H Hindi",
              "abl_slug": "lma-bonus-ablation-no-pos-model-h-hindi",
              "abl_title": "LMA Bonus Ablation No Pos Model H Hindi"},
    "nepali": {"slug": "lma-phase-2-model-l-nepali",
               "title": "LMA Phase 2 Model L Nepali",
               "dataset": "lma-nepali-packed",
               "eval_slug": "lma-phase-2-eval-model-l-nepali",
               "eval_title": "LMA Phase 2 Eval Model L Nepali",
               "ft_slug": "lma-phase-3-finetune-model-l-nepali",
               "ft_title": "LMA Phase 3 Finetune Model L Nepali",
               "abl_slug": "lma-bonus-ablation-no-pos-model-l-nepali",
               "abl_title": "LMA Bonus Ablation No Pos Model L Nepali"},
}


def slugify(title: str) -> str:
    """Reproduce Kaggle's title-to-slug rule.

    Args:
        title: The kernel title.

    Returns:
        The slug Kaggle will actually use.
    """
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")


# Which notebook, which kernel name, and what has to be attached, per mode.
# Phase 3 needs the pretraining kernel's output for the checkpoint, and needs
# only lma-code for data: the reasoning corpus ships with the code, so the
# multi-hundred-megabyte pretraining corpus is not mounted at all.
MODES = {
    "pretrain": {"slug": "slug", "title": "title",
                 "notebook": NOTEBOOK, "code_file": "kaggle_pretrain.ipynb",
                 "needs_corpus": True, "needs_pretrained_kernel": False},
    "eval": {"slug": "eval_slug", "title": "eval_title",
             "notebook": EVAL_NOTEBOOK, "code_file": "kaggle_evaluate.ipynb",
             "needs_corpus": True, "needs_pretrained_kernel": True},
    "finetune": {"slug": "ft_slug", "title": "ft_title",
                 "notebook": FINETUNE_NOTEBOOK,
                 "code_file": "kaggle_finetune.ipynb",
                 "needs_corpus": False, "needs_pretrained_kernel": True},
    # Bonus ablation: retrain from scratch with pos_encoding "none". It needs
    # the corpus like pretraining does, and explicitly does NOT attach the
    # pretraining kernel -- initialising from the standard model's weights
    # would defeat the entire point of the ablation.
    "ablation": {"slug": "abl_slug", "title": "abl_title",
                 "notebook": ABLATION_NOTEBOOK,
                 "code_file": "kaggle_ablation.ipynb",
                 "needs_corpus": True, "needs_pretrained_kernel": False},
}


for _lang, _spec in LANGUAGES.items():
    assert slugify(_spec["abl_title"]) == _spec["abl_slug"], (
        f"{_lang}: ablation title {_spec['abl_title']!r} slugifies to "
        f"{slugify(_spec['abl_title'])!r}, not {_spec['abl_slug']!r}"
    )
    assert slugify(_spec["ft_title"]) == _spec["ft_slug"], (
        f"{_lang}: finetune title {_spec['ft_title']!r} slugifies to "
        f"{slugify(_spec['ft_title'])!r}, not {_spec['ft_slug']!r}"
    )
    assert slugify(_spec["eval_title"]) == _spec["eval_slug"], (
        f"{_lang}: eval title {_spec['eval_title']!r} slugifies to "
        f"{slugify(_spec['eval_title'])!r}, not {_spec['eval_slug']!r}"
    )
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


def build_notebook(lang: str, notebook: Path = NOTEBOOK) -> dict:
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
    nb = json.loads(notebook.read_text(encoding="utf-8"))
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


def push(lang: str, dry_run: bool = False, mode: str = "pretrain") -> str:
    """Push the kernel for one language and start it running.

    Args:
        lang: Target language.
        dry_run: Stage the kernel directory but do not upload.
        mode: One of ``"pretrain"``, ``"eval"`` or ``"finetune"``.

    Returns:
        The kernel's full ``username/slug`` id.
    """
    spec = LANGUAGES[lang]
    mspec = MODES[mode]
    user = kaggle_username()
    slug = spec[mspec["slug"]]
    title = spec[mspec["title"]]
    notebook = mspec["notebook"]
    kernel_id = f"{user}/{slug}"

    with tempfile.TemporaryDirectory(prefix=f"kernel_{lang}_") as tmp:
        stage = Path(tmp)
        nb = build_notebook(lang, notebook)
        code_file = mspec["code_file"]
        (stage / code_file).write_text(
            json.dumps(nb, indent=1), encoding="utf-8"
        )
        metadata = {
            "id": kernel_id,
            "title": title,
            "code_file": code_file,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            # Request the T4 explicitly. Kaggle's default GPU is a P100, which
            # is Pascal (sm_60), and current PyTorch has dropped Pascal: the
            # card gets allocated and then every kernel launch fails with
            # "not compatible with the current PyTorch installation". The T4 is
            # sm_75 and supported. It also has tensor cores, so float16
            # training is faster on it than on the P100 regardless.
            "machine_shape": ACCELERATOR,
            "enable_internet": True,
            # Only this language's corpus is ever attached, so a notebook
            # training Model H is physically unable to read Model L's data.
            # Phase 3 skips the corpus entirely: the reasoning JSONL travels
            # inside lma-code, so mounting ~900 MB of pretraining tokens the run
            # never reads would only slow the kernel down.
            "dataset_sources": (
                [f"{user}/lma-code", f"{user}/{spec['dataset']}"]
                if mspec["needs_corpus"] else [f"{user}/lma-code"]
            ),
            "competition_sources": [],
            # Evaluation and finetuning both read the trained weights from the
            # pretraining kernel's saved output, so that run is attached as a
            # kernel source. Pretraining is the expensive part and is never
            # repeated; everything downstream reruns against the same weights.
            "kernel_sources": (
                [f"{user}/{spec['slug']}"]
                if mspec["needs_pretrained_kernel"] else []
            ),
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
        output = (result.stdout + result.stderr).strip()
        print(output)
        # The CLI exits 0 even when the server rejects the push - a rejected
        # push ("Maximum batch GPU session count of 2 reached") otherwise reads
        # as a successful launch. Trust the message, not the exit code.
        rejected = "error" in output.lower() and "successfully pushed" not in output
        if result.returncode != 0 or rejected:
            print(f"  push REJECTED - kernel was not started")
            raise RuntimeError(output.splitlines()[-1] if output else "push failed")
        print(f"  running -> https://www.kaggle.com/code/{kernel_id}")
    return kernel_id


def status(lang: str, mode: str = "pretrain") -> str:
    """Report the current status of one language's kernel.

    Args:
        lang: Target language.
        mode: One of ``"pretrain"``, ``"eval"`` or ``"finetune"``.

    Returns:
        The raw status string reported by Kaggle.
    """
    user = kaggle_username()
    kernel_id = f"{user}/{LANGUAGES[lang][MODES[mode]['slug']]}"
    result = subprocess.run(
        kaggle_cli() + ["kernels", "status", kernel_id],
        capture_output=True, text=True,
    )
    text = (result.stdout + result.stderr).strip()
    print(f"[{lang}] {text}")
    return text


def fetch(lang: str, out_dir: Path | None = None,
          mode: str = "pretrain") -> None:
    """Download a finished kernel's output files.

    Args:
        lang: Target language.
        out_dir: Destination directory; defaults to ``kaggle_output/<lang>/``.
        mode: One of ``"pretrain"``, ``"eval"`` or ``"finetune"``.
    """
    user = kaggle_username()
    kernel_id = f"{user}/{LANGUAGES[lang][MODES[mode]['slug']]}"
    dest = out_dir or (REPO_ROOT / "kaggle_output" / mode / lang)
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
    parser.add_argument(
        "--mode",
        default="pretrain",
        choices=sorted(MODES),
        help="pretrain (Phase 2), eval (Phase 2 scoring), or finetune "
        "(Phase 3 reasoning finetuning, evaluation and attention).",
    )
    parser.add_argument(
        "--eval",
        dest="evaluate",
        action="store_true",
        help="Deprecated alias for --mode eval.",
    )
    args = parser.parse_args()

    mode = "eval" if args.evaluate else args.mode

    if args.status:
        status(args.lang, mode=mode)
    elif args.fetch:
        fetch(args.lang, mode=mode)
    else:
        push(args.lang, dry_run=args.dry_run, mode=mode)


if __name__ == "__main__":
    main()
