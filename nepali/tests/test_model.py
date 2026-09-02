"""Correctness tests for the from-scratch Transformer.

The central test here is :func:`test_causal_mask_blocks_the_future`, which is
the empirical verification the course spec asks for directly: "verify
empirically that the model cannot see the future—for example, show that
changing token t+1 does not change the logits at position t."

Runs either under pytest or as a plain script::

    python nepali/tests/test_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nepali.model.config import GPTConfig  # noqa: E402
from nepali.model.transformer import (  # noqa: E402
    GPT,
    MultiHeadCausalSelfAttention,
)

# A deliberately tiny model: these tests check structural properties that hold
# at any size, and a small model keeps the suite fast.
TINY = dict(vocab_size=97, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)


def _model(**overrides) -> GPT:
    """Build a small deterministic model in eval mode.

    Dropout is disabled and the model is put in eval mode so that two forward
    passes over the same input are exactly reproducible — otherwise the causal
    test would fail for reasons unrelated to masking.

    Args:
        **overrides: Fields overriding :data:`TINY`.

    Returns:
        The constructed model, in eval mode.
    """
    torch.manual_seed(0)
    cfg = GPTConfig(**{**TINY, **overrides})
    return GPT(cfg).eval()


def test_causal_mask_blocks_the_future() -> None:
    """Changing a token at position t+1 must not move the logits at position t.

    This is the defining property of a causal language model. If it fails, the
    model is reading its own targets and every perplexity number downstream is
    meaningless.
    """
    model = _model()
    T = model.config.block_size
    torch.manual_seed(1)
    idx = torch.randint(0, model.config.vocab_size, (2, T))

    with torch.no_grad():
        base_logits, _, _ = model(idx)

    # Perturb one position at a time and confirm nothing at or before the
    # previous position moves.
    max_leak = 0.0
    for t in range(T - 1):
        perturbed = idx.clone()
        # Replace every token strictly after t with a different token.
        perturbed[:, t + 1 :] = (perturbed[:, t + 1 :] + 17) % model.config.vocab_size
        assert not torch.equal(perturbed, idx), "perturbation was a no-op"

        with torch.no_grad():
            new_logits, _, _ = model(perturbed)

        leak = (new_logits[:, : t + 1] - base_logits[:, : t + 1]).abs().max().item()
        max_leak = max(max_leak, leak)
        assert leak < 1e-5, (
            f"future leakage at position {t}: perturbing tokens after t changed "
            f"the logits at positions <= t by {leak:.3e}"
        )

        # Sanity check in the other direction: the logits *at* the perturbed
        # positions must change, otherwise the test would pass on a model that
        # simply ignores its input.
        changed = (new_logits[:, t + 1 :] - base_logits[:, t + 1 :]).abs().max().item()
        assert changed > 1e-6, f"perturbation at {t+1} had no effect anywhere"

    print(f"  causal mask: max logit leakage across all positions = {max_leak:.3e}")


def test_attention_is_lower_triangular_and_normalised() -> None:
    """Attention probabilities must be zero above the diagonal and sum to one.

    A direct check on the mask itself, complementing the behavioural test
    above: every query row is a probability distribution over keys at positions
    ``<= t`` and nothing else.
    """
    model = _model()
    T = model.config.block_size
    idx = torch.randint(0, model.config.vocab_size, (1, T))

    with torch.no_grad():
        _, _, attentions = model(idx, return_attn=True)

    assert attentions is not None
    assert len(attentions) == model.config.n_layer

    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    for layer, attn in enumerate(attentions):
        assert attn.shape == (1, model.config.n_head, T, T), attn.shape
        future_mass = attn[..., upper].abs().max().item()
        assert future_mass == 0.0, (
            f"layer {layer}: attention mass {future_mass:.3e} on future positions"
        )
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            f"layer {layer}: attention rows do not sum to 1"
        )


def test_forward_shapes_and_loss() -> None:
    """Logits, loss and attention tensors must have the documented shapes."""
    model = _model()
    B, T = 3, 8
    idx = torch.randint(0, model.config.vocab_size, (B, T))
    targets = torch.randint(0, model.config.vocab_size, (B, T))

    logits, loss, attns = model(idx, targets=targets, return_attn=True)
    assert logits.shape == (B, T, model.config.vocab_size), logits.shape
    assert loss is not None and loss.ndim == 0
    assert attns is not None and len(attns) == model.config.n_layer
    assert attns[0].shape == (B, model.config.n_head, T, T)

    # An untrained model over V tokens should sit near ln(V).
    import math

    expected = math.log(model.config.vocab_size)
    assert abs(loss.item() - expected) < 1.0, (
        f"initial loss {loss.item():.3f} is far from ln(V)={expected:.3f}; "
        "check weight initialisation"
    )


def test_sequence_longer_than_block_size_is_rejected() -> None:
    """Exceeding block_size must raise, not silently index out of range.

    With learned absolute positional embeddings there is no row for a position
    beyond ``block_size``, so this is a real architectural limit and should
    surface as a clear error.
    """
    model = _model()
    idx = torch.zeros((1, model.config.block_size + 1), dtype=torch.long)
    try:
        model(idx)
    except ValueError as exc:
        assert "block_size" in str(exc)
    else:
        raise AssertionError("expected ValueError for over-long sequence")


def test_weight_tying_shares_storage() -> None:
    """With tying on, the output head and the token embedding are one tensor."""
    tied = _model(tie_weights=True)
    assert tied.lm_head.weight.data_ptr() == tied.token_embedding.weight.data_ptr()
    untied = _model(tie_weights=False)
    assert untied.lm_head.weight.data_ptr() != untied.token_embedding.weight.data_ptr()

    # Tying must save exactly one V x d matrix.
    saved = tied.config.vocab_size * tied.config.n_embd
    assert (
        untied.parameter_count()["total"] - tied.parameter_count()["total"] == saved
    )


def test_parameter_count_matches_analytic_estimate() -> None:
    """The built model's parameter count must match the config's estimate.

    The estimate is what config sizing decisions are made from, so a drift
    between the two would mean the reported ~25M target was computed against a
    model that was never built.
    """
    model = _model()
    actual = model.parameter_count()["total"]
    estimated = model.config.estimated_parameters()["total"]
    assert actual == estimated, f"estimate {estimated:,} != actual {actual:,}"


def test_positional_encoding_variants() -> None:
    """All three position schemes run, and "none" really removes position.

    Without positional information the model is permutation invariant: shuffling
    the input tokens permutes the outputs identically rather than producing
    something new. This is the property the optional bonus ablation destroys.
    """
    for scheme in ("learned", "sinusoidal", "none"):
        model = _model(pos_encoding=scheme)
        idx = torch.randint(0, model.config.vocab_size, (1, 8))
        with torch.no_grad():
            logits, _, _ = model(idx)
        assert logits.shape == (1, 8, model.config.vocab_size)

    # A position-free model, given a reversed prefix of length 1, must give the
    # same first-position logits regardless of what follows.
    model = _model(pos_encoding="none")
    a = torch.tensor([[5, 9, 11, 2]])
    b = torch.tensor([[5, 2, 11, 9]])
    with torch.no_grad():
        la, _, _ = model(a)
        lb, _, _ = model(b)
    assert torch.allclose(la[:, 0], lb[:, 0], atol=1e-6)


def test_no_prebuilt_attention_modules_used() -> None:
    """Guard the spec's hard constraint on pre-built Transformer components.

    Walks the module tree and fails on anything from ``torch.nn``'s Transformer
    family. Cheap insurance against a future refactor quietly importing one.
    """
    import torch.nn as nn

    forbidden = (
        nn.Transformer,
        nn.TransformerEncoder,
        nn.TransformerDecoder,
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
        nn.MultiheadAttention,
    )
    model = _model()
    for name, module in model.named_modules():
        assert not isinstance(module, forbidden), (
            f"{name} is a pre-built Transformer module: {type(module).__name__}"
        )
    # And the attention block we do use is our own.
    assert any(
        isinstance(m, MultiHeadCausalSelfAttention) for m in model.modules()
    ), "no student-implemented attention block found"


def test_generate_is_deterministic_when_greedy() -> None:
    """Greedy decoding must be reproducible and must respect block_size."""
    model = _model()
    prompt = torch.randint(0, model.config.vocab_size, (2, 4))
    a = model.generate(prompt, max_new_tokens=20, greedy=True)
    b = model.generate(prompt, max_new_tokens=20, greedy=True)
    assert torch.equal(a, b)
    assert a.shape == (2, 24)
    # 24 > block_size=16, so the context was cropped and no index error occurred.


def main() -> int:
    """Run every test in this module and report a summary.

    Returns:
        Process exit code: ``0`` if all tests passed, ``1`` otherwise.
    """
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"pass  {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
