"""Generate ``notebooks/kaggle_finetune.ipynb``, the Phase 3 Kaggle kernel.

The notebook is generated rather than hand-edited for one specific reason.
nbformat stores a cell's source as a *list of lines*, and every element except
the last must carry its own trailing newline. A file that omits them looks
perfectly valid as JSON and then collapses into one unbroken line when Jupyter
joins it, producing a SyntaxError on a machine you are not sitting at. That
failure cost a GPU session during Phase 2. :func:`lines` below is the fix, and
the generator re-validates its own output by parsing each cell exactly as
Jupyter will join it.

Usage
-----
    python scripts/make_finetune_notebook.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "notebooks/kaggle_finetune.ipynb"


def lines(text: str) -> list[str]:
    """Split source into nbformat's line list, keeping trailing newlines.

    Args:
        text: The cell source as one string.

    Returns:
        One list element per line, each but the last ending in ``\\n``.
    """
    parts = text.strip("\n").split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(text: str) -> dict:
    """Build a markdown cell.

    Args:
        text: Markdown source.

    Returns:
        The nbformat cell dict.
    """
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    """Build a code cell.

    Args:
        text: Python source.

    Returns:
        The nbformat cell dict.
    """
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": lines(text),
    }


INTRO = """
# LMA Phase 3 - reasoning finetuning, evaluation and attention

One kernel covers four of the six Phase 3 deliverables for a single model:

1. **Pretrained baseline** on the reasoning test set, scored *before* any
   finetuning. "Pretrained vs. finetuned accuracy" is meaningless without it,
   and once the weights are updated the baseline cannot be recovered.
2. **Finetuning** from that language's own Phase 2 checkpoint, with its Phase 1
   tokenizer held fixed.
3. **Finetuned evaluation**, through the exact same code path as the baseline.
4. **Attention comparison**, pretrained vs. finetuned, one early and one late
   layer, on held-out comparative-reasoning prompts.

### Before running
Attach, as inputs:

* the `lma-code` dataset (code, reasoning corpora, tokenizers), and
* this language's Phase 2 **pretraining kernel output**, which holds the
  checkpoint.

The pretraining corpus is deliberately *not* required: finetuning reads only the
reasoning JSONL, which ships inside `lma-code`.
"""

ENVIRONMENT = '''
import os, sys, json, shutil, zipfile, time
from pathlib import Path
import torch

print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU  :", torch.cuda.get_device_name(0),
          "| cc %d.%d" % (props.major, props.minor))
    # Kaggle's default GPU is a P100 (sm_60), which current PyTorch has dropped:
    # the card allocates fine and then every kernel launch fails. Fail loudly
    # here rather than an hour into a run.
    if (props.major, props.minor) < (7, 0):
        raise RuntimeError("compute capability below 7.0; request NvidiaTeslaT4")
    torch.zeros(8, 8, device="cuda").matmul(torch.zeros(8, 8, device="cuda"))
    print("     matmul probe OK")

INPUT_ROOT = Path("/kaggle/input")


def walk_dirs(root, max_depth=6):
    # Kaggle nests mounts and does not name them after the dataset slug, so
    # everything below is located by content rather than by path.
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        yield current
        if depth < max_depth:
            try:
                for child in sorted(current.iterdir()):
                    if child.is_dir():
                        stack.append((child, depth + 1))
            except OSError:
                pass


REPO = Path("/kaggle/working/repo")
if REPO.exists():
    shutil.rmtree(REPO)
REPO.mkdir(parents=True)

# The attached pretraining run copied its own repo/ into its output, so more
# than one mount can look like the code tree. Prefer the lma-code dataset:
# picking the stale copy embedded in a previous run's output would silently run
# Phase 3 against a revision that predates it.
candidates = [c for c in walk_dirs(INPUT_ROOT)
              if (c / "hindi").is_dir() and (c / "nepali").is_dir()]


def code_rank(path):
    text = str(path)
    if "lma-code" in text:
        return 0
    if "/datasets/" in text:
        return 1
    return 2


candidates.sort(key=lambda c: (code_rank(c), len(str(c))))
if not candidates:
    raise FileNotFoundError("code dataset not found among attached inputs")
src = candidates[0]
print("code candidates:", [str(c) for c in candidates])
for item in sorted(src.iterdir()):
    target = REPO / item.name
    if item.is_dir():
        shutil.copytree(item, target)
    else:
        shutil.copy2(item, target)
sys.path.insert(0, str(REPO))
os.chdir(REPO)
print("code from", src)
'''

LOCATE_MD = """
## 2. Locate this language's pretrained checkpoint

The checkpoint is identified by the language recorded *inside* it, not by its
path. Mounting the other model's weights would produce a complete set of
plausible numbers for the wrong model, and nothing downstream would notice.
"""

LOCATE = '''
# ---- the only setting you change between the two runs ----
LANGUAGE = "hindi"          # "hindi" (Model H) or "nepali" (Model L)
PRETRAINED_NAME = "ckpt_best.pt"
# ----------------------------------------------------------
import yaml

found = [c / PRETRAINED_NAME for c in walk_dirs(INPUT_ROOT)
         if (c / PRETRAINED_NAME).is_file()]
# Checking the likely directory first only saves time; correctness comes from
# the config check below, which every candidate has to pass.
found.sort(key=lambda p: 0 if p.parent.name == LANGUAGE else 1)

CKPT = None
for hit in found:
    blob = torch.load(hit, map_location="cpu", weights_only=False)
    recorded = blob.get("config", {}).get("train", {}).get("language")
    print("  candidate", hit, "-> language:", recorded, "| step", blob.get("step"))
    if recorded == LANGUAGE:
        CKPT = hit
        break
    del blob
if CKPT is None:
    raise FileNotFoundError(
        "no attached input holds a " + LANGUAGE + " pretrained checkpoint; "
        "attach the Phase 2 pretraining kernel output")

# Model H trained three candidate vocabularies in Phase 1 and kept the 8k one,
# so the tokenizer directory holds more than one model file. The config names
# the one that was actually used for pretraining, and the finetuner asserts the
# vocabulary size matches the checkpoint.
ft_yaml = yaml.safe_load(
    (REPO / LANGUAGE / "configs" / "finetune.yaml").read_text(encoding="utf-8"))
TOKENIZER = REPO / ft_yaml["tokenizer_model"]
if not TOKENIZER.is_file():
    raise FileNotFoundError("tokenizer named by config is missing: "
                            + str(TOKENIZER))

OUT = Path("/kaggle/working/phase3")
FIGS = OUT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

print()
print("language  :", LANGUAGE)
print("checkpoint:", CKPT, "(%.0f MB)" % (CKPT.stat().st_size / 1e6))
print("tokenizer :", TOKENIZER)
'''

BASELINE_MD = """
## 3. Pretrained baseline - **must run before finetuning**

Two metrics, for the reason set out in `evaluate_reasoning.py`: greedy exact
match is the honest end-to-end number but floors near zero on a model that has
never seen this prompt format, while forced choice among the entities the prompt
names has a known chance rate and so stays interpretable.
"""

BASELINE = '''
import importlib
ev_mod = importlib.import_module(LANGUAGE + ".reasoning.evaluate_reasoning")


def score(checkpoint, tag):
    """Score one checkpoint on the reasoning test split and save the report."""
    started = time.time()
    ev = ev_mod.ReasoningEvaluator(str(checkpoint), tokenizer_model=TOKENIZER)
    data = ev.load_split("test")
    report = ev_mod.build_report(ev, data, tag, "test", batch_size=64)
    path = OUT / (LANGUAGE + "_reasoning_" + tag + ".json")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print("  %-10s exact match %6.2f%%   forced choice %6.2f%% (chance %5.2f%%)"
          "   answer loss %.4f   [%.1f min]"
          % (tag, 100 * report["exact_match"]["accuracy"],
             100 * report["forced_choice"]["accuracy"],
             100 * report["forced_choice"]["chance_accuracy"],
             report["answer_token_loss"]["answer_token_loss"],
             (time.time() - started) / 60))
    del ev
    torch.cuda.empty_cache()
    return report


baseline = score(CKPT, "pretrained")
'''

FINETUNE_MD = """
## 4. Finetune

Starts from the checkpoint located above and nothing else. Hyperparameters come
from `<language>/configs/finetune.yaml` and are identical across the two models,
so any difference in reasoning results is attributable to data and tokenizer
rather than to the finetuning recipe.
"""

FINETUNE = '''
ft_mod = importlib.import_module(LANGUAGE + ".reasoning.finetune")

cfg = ft_mod.FinetuneConfig.load(REPO / LANGUAGE / "configs" / "finetune.yaml")
cfg.pretrained_checkpoint = str(CKPT)
cfg.tokenizer_model = str(TOKENIZER)
cfg.out_dir = "checkpoints/" + LANGUAGE + "_reasoning"
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

started = time.time()
tuner = ft_mod.FineTuner(cfg, resume=True)
tuner.train()
print("finetuned in %.1f min" % ((time.time() - started) / 60))

FT_DIR = REPO / cfg.out_dir
FT_BEST = FT_DIR / "ckpt_best.pt"
print("checkpoints:", sorted(p.name for p in FT_DIR.glob("*.pt")))
'''

TUNED_MD = "## 5. Finetuned evaluation - same code path as the baseline"

TUNED = '''
tuned = score(FT_BEST, "finetuned")

print()
print("%-14s %12s %12s" % ("metric", "pretrained", "finetuned"))
print("-" * 40)
print("%-14s %11.2f%% %11.2f%%" % ("exact match",
      100 * baseline["exact_match"]["accuracy"],
      100 * tuned["exact_match"]["accuracy"]))
print("%-14s %11.2f%% %11.2f%%" % ("forced choice",
      100 * baseline["forced_choice"]["accuracy"],
      100 * tuned["forced_choice"]["accuracy"]))
print("%-14s %12.4f %12.4f" % ("answer loss",
      baseline["answer_token_loss"]["answer_token_loss"],
      tuned["answer_token_loss"]["answer_token_loss"]))
print()
for task in sorted(tuned["exact_match"]["by_task"]):
    print("  %-12s EM %6.2f%% -> %6.2f%%   FC %6.2f%% -> %6.2f%%" % (
        task,
        100 * baseline["exact_match"]["by_task"][task]["accuracy"],
        100 * tuned["exact_match"]["by_task"][task]["accuracy"],
        100 * baseline["forced_choice"]["by_task"][task]["accuracy"],
        100 * tuned["forced_choice"]["by_task"][task]["accuracy"]))
'''

ATTENTION_MD = """
## 6. Attention: pretrained vs. finetuned

Both checkpoints are run on the **same** held-out comparative-reasoning prompts,
all of one token length, so neither the choice of text nor the sequence length
can masquerade as a finetuning effect.
"""

ATTENTION = '''
import subprocess
subprocess.run("apt-get -qq install -y fonts-noto-devanagari > /dev/null 2>&1",
               shell=True)
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)

ac_mod = importlib.import_module(LANGUAGE + ".reasoning.attention_compare")
comparison = ac_mod.compare(
    CKPT, FT_BEST, tokenizer_model=TOKENIZER, split="test",
    batch_size=16, fig_dir=FIGS,
)
(OUT / (LANGUAGE + "_attention_comparison.json")).write_text(
    json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

print("prompts compared:", comparison["prompts_compared"],
      "| T =", comparison["sequence_length"],
      "| uniform baseline entropy %.3f" % comparison["uniform_baseline_entropy"])
for name, block in comparison["models"].items():
    print("  %-11s mean entropy %.4f   mean distance %.3f"
          % (name, block["mean_entropy"], block["mean_distance"]))
print()
print("heads that moved most (entropy):")
for head, delta in comparison["largest_entropy_shifts"][:6]:
    print("  %-6s %+.4f nats   %+.3f positions"
          % (head, delta["entropy_delta"], delta["distance_delta"]))
print()
print("example prompt:", comparison["example_prompts"][0])
'''

BUNDLE_MD = "## 7. Bundle for the repository and for Drive"

BUNDLE = '''
# Reports and figures go into the git branch; the spec is explicit that a grader
# will not open Drive to find a figure.
bundle = Path("/kaggle/working/report_phase3_" + LANGUAGE + ".zip")
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUT.glob("*.json")):
        zf.write(p, p.name)
    for p in sorted(FIGS.glob("*.png")):
        zf.write(p, "figures/" + p.name)
    log = FT_DIR / "finetune_log.jsonl"
    if log.is_file():
        zf.write(log, LANGUAGE + "_finetune_log.jsonl")
    conf = FT_DIR / "finetune_config.yaml"
    if conf.is_file():
        zf.write(conf, LANGUAGE + "_finetune_config.yaml")
print("%-44s %6.2f MB" % (bundle.name, bundle.stat().st_size / 1e6))

# Checkpoints are large binaries and belong on Drive, not in git.
ckpt_zip = Path("/kaggle/working/model_"
                + ("H" if LANGUAGE == "hindi" else "L") + "_" + LANGUAGE
                + "_reasoning_finetuned.zip")
with zipfile.ZipFile(ckpt_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(FT_DIR.glob("*.pt")):
        zf.write(p, p.name)
    for p in sorted(FT_DIR.glob("*.yaml")):
        zf.write(p, p.name)
    for p in sorted(FT_DIR.glob("*.jsonl")):
        zf.write(p, p.name)
print("%-44s %6.2f MB  -> upload to Drive"
      % (ckpt_zip.name, ckpt_zip.stat().st_size / 1e6))
'''


def build() -> dict:
    """Assemble the notebook.

    Returns:
        The notebook as an nbformat dict.
    """
    cells = [
        md(INTRO),
        md("## 1. Environment"),
        code(ENVIRONMENT),
        md(LOCATE_MD),
        code(LOCATE),
        md(BASELINE_MD),
        code(BASELINE),
        md(FINETUNE_MD),
        code(FINETUNE),
        md(TUNED_MD),
        code(TUNED),
        md(ATTENTION_MD),
        code(ATTENTION),
        md(BUNDLE_MD),
        code(BUNDLE),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    """Write the notebook and validate it the way the pusher will."""
    nb = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)} with {len(nb['cells'])} cells")

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        ast.parse("".join(cell["source"]))  # exactly how Jupyter joins it
    print("every code cell parses as Jupyter will join it")

    assignments = sum(
        1
        for cell in nb["cells"] if cell["cell_type"] == "code"
        for line in cell["source"] if line.startswith('LANGUAGE = "')
    )
    if assignments != 1:
        raise RuntimeError(
            f"kaggle_run.build_notebook requires exactly one LANGUAGE "
            f"assignment; found {assignments}"
        )
    print("LANGUAGE assignments: 1")


if __name__ == "__main__":
    main()
