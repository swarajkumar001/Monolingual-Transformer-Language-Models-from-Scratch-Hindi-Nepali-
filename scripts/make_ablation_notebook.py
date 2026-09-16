"""Generate the bonus-ablation Kaggle notebook from the pretraining one.

The ablation must run through *exactly* the same code path as the standard
model, or the comparison is worthless. Rather than hand-maintaining a second
notebook that could drift, this derives it from ``kaggle_pretrain.ipynb`` and
patches only the three lines that must differ:

    CONFIG    -> hindi/configs/model_nopos.yaml   (pos_encoding: none)
    OUT_DIR   -> /kaggle/working/checkpoints/hindi_nopos
    MAX_MINUTES -> a shorter wall-clock budget

Everything else -- staging, the independence guard, the smoke test, training,
the full Phase 2 evaluation suite, attention collection and the figures -- is
inherited unchanged.

Run::

    python scripts/make_ablation_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "notebooks" / "kaggle_pretrain.ipynb"
TARGET = REPO_ROOT / "notebooks" / "kaggle_ablation.ipynb"

# Wall-clock budget for training, in minutes. The notebook still has to run the
# evaluation suite and attention analysis after training inside the same
# session, so this is deliberately well short of Kaggle's 12-hour cap.
MAX_MINUTES = 330

REPLACEMENTS = [
    (
        'MAX_MINUTES = 660           # stop cleanly at 11h, inside Kaggle\'s 12h cap\n',
        f"MAX_MINUTES = {MAX_MINUTES}           "
        "# shorter than the baseline's 11h: this is a step-matched ablation\n",
    ),
    (
        'CONFIG    = REPO / (LANGUAGE + "/configs/model.yaml")\n',
        'CONFIG    = REPO / (LANGUAGE + "/configs/model_nopos.yaml")\n',
    ),
    (
        'OUT_DIR   = Path("/kaggle/working/checkpoints/" + LANGUAGE)\n',
        'OUT_DIR   = Path("/kaggle/working/checkpoints/" + LANGUAGE + "_nopos")\n',
    ),
    (
        'print("report_phase2.zip  %.1f MB  -> commit into the repo"\n',
        'print("report_ablation.zip  %.1f MB  -> commit into the repo"\n',
    ),
    (
        'small = Path("/kaggle/working/report_phase2.zip")\n',
        'small = Path("/kaggle/working/report_ablation.zip")\n',
    ),
]

HEADER = """# LMA Individual Project - BONUS ABLATION: no positional embeddings

Retrains **Model H (Hindi)** with `pos_encoding: none`, then runs the full
Phase 2 evaluation suite on it, so the ablated model can be compared against the
standard Model H directly.

This notebook is **generated** from `kaggle_pretrain.ipynb` by
`scripts/make_ablation_notebook.py`. It is the same code path as the standard
run; only the config, the output directory and the wall-clock budget differ.

### Why Hindi

Its tokenizer has byte fallback and a measured 0.000000 unknown-token rate, so
no part of any observed difference can be blamed on `<unk>` noise.

### What is being tested

Self-attention is permutation invariant: it treats its input as a set. Without a
positional signal the model cannot tell "A is bigger than B" from "B is bigger
than A". This run measures what that costs on a real corpus.

### Before running

1. **Accelerator:** GPU T4 x2 is fine, the code uses one card.
2. **Attach datasets** (Add Input): `swaraj001/lma-code` and
   `swaraj001/lma-hindi-packed`. Do **not** attach the Nepali corpus - the
   notebook fails loudly if you do.
3. **Internet:** on.
"""


def main() -> None:
    """Patch the pretraining notebook and write the ablation notebook."""
    sys.stdout.reconfigure(encoding="utf-8")

    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))

    applied = {old: 0 for old, _ in REPLACEMENTS}
    for cell in notebook["cells"]:
        source = cell["source"]
        for index, line in enumerate(source):
            for old, new in REPLACEMENTS:
                if line == old:
                    source[index] = new
                    applied[old] += 1

    missing = [old.strip()[:60] for old, count in applied.items() if count == 0]
    if missing:
        raise SystemExit(
            "these lines were not found in the source notebook, so the "
            "ablation notebook would be wrong:\n  " + "\n  ".join(missing)
        )

    # Replace the title cell wholesale rather than patching its prose.
    notebook["cells"][0]["source"] = [l + "\n" for l in HEADER.splitlines()]

    # Every code cell must still parse, for the same reason the finetune
    # notebook generator checks: a notebook that fails at cell 18 has already
    # burned the GPU hours spent getting there.
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<cell>", "exec")

    TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8")

    for old, count in applied.items():
        print(f"  patched {count}x  {old.strip()[:66]}")
    print(f"\n  MAX_MINUTES = {MAX_MINUTES}")
    print(f"  wrote {TARGET.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
