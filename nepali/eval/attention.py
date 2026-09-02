"""Attention analysis: heatmaps, per-head entropy, and mean attention distance.

Because the attention block is student-implemented, its post-softmax probability
matrices are available directly rather than having to be hooked out of a library
module. Every statistic here is computed over the *causally valid* region only:
query position ``i`` has exactly ``i + 1`` reachable keys, and averaging over the
masked upper triangle would otherwise wash the numbers out with structural zeros.

Phase 3 reuses this module unchanged on the finetuned checkpoints, so pretrained
and finetuned attention can be compared on identical footing.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..model.transformer import GPT


def collect_attention(
    model: GPT, token_ids: torch.Tensor
) -> list[torch.Tensor]:
    """Run the model once and return the attention probabilities of every layer.

    Args:
        model: The model to inspect.
        token_ids: Token ids of shape ``(B, T)``.

    Returns:
        A list of ``n_layer`` tensors, each ``(B, h, T, T)`` on CPU in float32.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, _, attentions = model(token_ids, return_attn=True)
    if was_training:
        model.train()
    assert attentions is not None
    return [a.detach().float().cpu() for a in attentions]


def attention_entropy(attn: torch.Tensor, normalise: bool = False) -> np.ndarray:
    """Compute the entropy of each head's attention distribution.

    Entropy is measured in nats over the key axis and then averaged across query
    positions and batch. A head with high entropy spreads its probability mass
    broadly (a diffuse, averaging head); a head with low entropy concentrates on
    a few positions (a sharp, selective head).

    Query position 0 is skipped: it can only attend to itself, so its entropy is
    identically zero and including it would bias every head downward by the same
    uninformative amount.

    Args:
        attn: Attention probabilities of shape ``(B, h, T, T)``.
        normalise: If True, divide each query's entropy by ``log(i + 1)``, the
            maximum achievable entropy given that position ``i`` has ``i + 1``
            reachable keys. This removes the artefact that early positions
            *cannot* be diffuse, making heads comparable across the sequence.

    Returns:
        Array of shape ``(h,)`` giving mean entropy per head.
    """
    B, h, T, _ = attn.shape
    eps = 1e-12
    # -sum p log p over the key axis -> (B, h, T)
    ent = -(attn * (attn + eps).log()).sum(dim=-1)
    if normalise:
        # log(i + 1) is the entropy of a uniform distribution over i+1 keys.
        max_ent = torch.log(
            torch.arange(1, T + 1, dtype=ent.dtype).clamp(min=1)
        ).clamp(min=eps)
        ent = ent / max_ent.view(1, 1, T)
    return ent[:, :, 1:].mean(dim=(0, 2)).numpy()


def mean_attention_distance(attn: torch.Tensor) -> np.ndarray:
    """Compute how far back, on average, each head attends.

    For query ``i``, the expected distance is ``sum_j p[i, j] * (i - j)``, which
    is zero for a head that only looks at the current token and grows for heads
    that reach back into the context. Contrasting this across heads is what
    separates local/positional heads from long-range, content-based ones.

    Query position 0 is skipped, since it has no history and contributes a
    forced zero.

    Args:
        attn: Attention probabilities of shape ``(B, h, T, T)``.

    Returns:
        Array of shape ``(h,)`` giving mean attention distance per head, in
        token positions.
    """
    B, h, T, _ = attn.shape
    positions = torch.arange(T, dtype=attn.dtype)
    # distance[i, j] = i - j, non-negative on the causal (lower) triangle.
    distance = positions.view(T, 1) - positions.view(1, T)
    expected = (attn * distance.view(1, 1, T, T)).sum(dim=-1)  # (B, h, T)
    return expected[:, :, 1:].mean(dim=(0, 2)).numpy()


def attention_to_self(attn: torch.Tensor) -> np.ndarray:
    """Compute each head's average probability mass on the current token.

    A head with high self-attention is largely passing its own representation
    through, which often marks a head that has specialised into an identity or
    "no-op" role.

    Args:
        attn: Attention probabilities of shape ``(B, h, T, T)``.

    Returns:
        Array of shape ``(h,)``.
    """
    T = attn.shape[-1]
    diag = attn[..., torch.arange(T), torch.arange(T)]  # (B, h, T)
    return diag[:, :, 1:].mean(dim=(0, 2)).numpy()


def summarise_model_attention(
    model: GPT, token_ids: torch.Tensor, normalise_entropy: bool = True
) -> dict:
    """Compute per-layer, per-head attention statistics for one input batch.

    Args:
        model: The model to inspect.
        token_ids: Token ids of shape ``(B, T)``.
        normalise_entropy: Whether to report length-normalised entropy
            alongside the raw value.

    Returns:
        A dict with ``n_layer``, ``n_head``, ``sequence_length`` and per-layer
        lists of ``entropy``, ``entropy_normalised``, ``mean_distance`` and
        ``self_attention``, plus whole-model means.
    """
    attentions = collect_attention(model, token_ids)
    layers = []
    for layer_idx, attn in enumerate(attentions):
        layers.append(
            {
                "layer": layer_idx,
                "entropy": attention_entropy(attn).tolist(),
                "entropy_normalised": (
                    attention_entropy(attn, normalise=True).tolist()
                    if normalise_entropy
                    else None
                ),
                "mean_distance": mean_attention_distance(attn).tolist(),
                "self_attention": attention_to_self(attn).tolist(),
            }
        )
    all_entropy = np.array([l["entropy"] for l in layers])
    all_distance = np.array([l["mean_distance"] for l in layers])
    return {
        "n_layer": len(attentions),
        "n_head": attentions[0].shape[1],
        "sequence_length": attentions[0].shape[-1],
        "layers": layers,
        "mean_entropy": float(all_entropy.mean()),
        "mean_distance": float(all_distance.mean()),
        # A head is called "local" when it attends, on average, within a short
        # window; the threshold is a reporting convention, not a property of the
        # model, and is stated explicitly so the number can be reproduced.
        "local_head_threshold": 8.0,
        "local_head_fraction": float((all_distance < 8.0).mean()),
    }


def head_classification(summary: dict) -> list[dict]:
    """Label each head as local, mid-range or long-range with a sharpness note.

    A descriptive convenience for the report's discussion of which heads look
    positional and which look content-based; the thresholds are reporting
    conventions and are returned alongside the labels.

    Args:
        summary: The output of :func:`summarise_model_attention`.

    Returns:
        One record per head, with its layer, head index, distance, entropy and
        assigned labels.
    """
    records = []
    all_entropy = np.array([l["entropy"] for l in summary["layers"]])
    median_entropy = float(np.median(all_entropy))
    for layer in summary["layers"]:
        for head_idx, (dist, ent) in enumerate(
            zip(layer["mean_distance"], layer["entropy"])
        ):
            if dist < 4:
                span = "local"
            elif dist < 32:
                span = "mid-range"
            else:
                span = "long-range"
            records.append(
                {
                    "layer": layer["layer"],
                    "head": head_idx,
                    "mean_distance": dist,
                    "entropy": ent,
                    "span": span,
                    "sharpness": "sharp" if ent < median_entropy else "diffuse",
                }
            )
    return records


def uniform_baseline_entropy(seq_len: int) -> float:
    """Return the mean entropy a perfectly uniform causal head would have.

    This is the reference point every measured entropy should be read against:
    a head at this value is doing nothing but averaging its context, while a
    head far below it has learned to select.

    Args:
        seq_len: Sequence length.

    Returns:
        Mean of ``log(i + 1)`` over query positions ``i >= 1``, in nats.
    """
    return float(np.mean([math.log(i + 1) for i in range(1, seq_len)]))
