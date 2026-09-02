"""A decoder-only (GPT-style) Transformer language model, written from scratch.

Everything in this file is built from primitive PyTorch layers only —
``nn.Linear``, ``nn.Embedding``, ``nn.Dropout``, ``nn.Parameter`` and plain
tensor ops. No ``nn.Transformer*`` module, no ``nn.MultiheadAttention``, no
``F.scaled_dot_product_attention``, and no HuggingFace model class is used
anywhere. The attention scores, the causal mask and the softmax are computed
explicitly so that every tensor operation in the forward pass is visible, and
so that the attention probability matrices can be returned for the Phase 2/3
attention analysis.

Shape conventions used throughout:

    B  batch size
    T  sequence length (number of tokens), ``T <= config.block_size``
    d  model dimension (``config.n_embd``)
    h  number of attention heads (``config.n_head``)
    dk per-head dimension (``d / h``)
    V  vocabulary size
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class LayerNorm(nn.Module):
    """Layer normalisation with an optional bias term.

    ``nn.LayerNorm`` always allocates a bias when affine; this thin wrapper
    around ``F.layer_norm`` lets the bias be dropped, which is a common choice
    in modern GPT implementations and removes a small number of parameters that
    empirically do little.

    Normalisation is over the feature dimension only — each token position is
    normalised independently of the others, which is what keeps the operation
    compatible with causal masking (it never mixes information across time).
    """

    def __init__(self, ndim: int, bias: bool = True) -> None:
        """Create a LayerNorm over the last dimension.

        Args:
            ndim: Size of the normalised (feature) dimension.
            bias: Whether to learn an additive shift alongside the gain.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the final dimension of ``x``.

        Args:
            x: Tensor of shape ``(..., ndim)``.

        Returns:
            Tensor of the same shape, normalised then scaled and shifted.
        """
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, implemented from first principles.

    For an input ``X`` of shape ``(B, T, d)`` this computes, per head:

        Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(dk) + M ) V

    where ``Q = X W_Q``, ``K = X W_K``, ``V = X W_V``, and ``M`` is an additive
    causal mask holding ``0`` on allowed positions and ``-inf`` on future ones.

    Why the ``1/sqrt(dk)`` scaling: ``Q`` and ``K`` entries are roughly unit
    variance at initialisation, so a dot product over ``dk`` dimensions has
    variance on the order of ``dk``. Without rescaling, the logits fed to the
    softmax grow with ``dk``; the softmax then saturates, producing a
    near-one-hot distribution whose gradients vanish. Dividing by ``sqrt(dk)``
    restores unit-scale logits and keeps the softmax in its responsive regime.

    Why the mask: this is a *causal* language model, so the prediction at
    position ``t`` may depend only on positions ``<= t``. Setting future scores
    to ``-inf`` before the softmax drives their probabilities to exactly zero,
    so no information flows backwards in time. Removing the mask would let the
    model read the answer it is being trained to predict, and validation loss
    would collapse toward zero while the model learned nothing useful.

    The three projections are kept as separate ``nn.Linear`` modules rather than
    fused into one ``(d, 3d)`` matmul, so that ``W_Q``, ``W_K`` and ``W_V`` map
    one-to-one onto the mathematical description above.
    """

    def __init__(self, config: GPTConfig) -> None:
        """Build the projections, dropouts and causal mask buffer.

        Args:
            config: Architecture configuration.
        """
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim

        # Q, K, V projections: (B, T, d) -> (B, T, d) each.
        self.W_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.W_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.W_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Output projection applied after the heads are concatenated back to d.
        self.W_o = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Upper-triangular boolean mask, True where a position is in the future
        # and must be blocked. Registered as a non-persistent buffer: it moves
        # with the module across devices but is not written into checkpoints,
        # since it is fully determined by block_size.
        causal = torch.triu(
            torch.ones(config.block_size, config.block_size, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run masked multi-head self-attention over a sequence.

        Args:
            x: Input activations of shape ``(B, T, d)``.
            return_attn: If True, also return the post-softmax attention
                probabilities. Kept optional because materialising a
                ``(B, h, T, T)`` tensor is expensive and only the analysis code
                needs it.

        Returns:
            A ``(output, attn)`` pair. ``output`` has shape ``(B, T, d)``.
            ``attn`` is ``(B, h, T, T)`` when requested, else ``None``.

        Raises:
            ValueError: If ``T`` exceeds the configured ``block_size``, since
                the causal mask buffer is only defined up to that length.
        """
        B, T, d = x.shape
        if T > self.causal_mask.size(0):
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.causal_mask.size(0)}"
            )

        # (B, T, d) -> (B, T, d), one projection per role.
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Split the model dimension into h heads of width dk, then move the head
        # axis in front of time so each head is an independent (T, dk) sequence:
        #   (B, T, d) -> (B, T, h, dk) -> (B, h, T, dk)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product scores: (B, h, T, dk) @ (B, h, dk, T) -> (B, h, T, T).
        # Entry [b, head, i, j] is how much query position i attends to key j.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Additive causal mask: -inf on future positions, so softmax sends them
        # to exactly zero. Sliced to T so shorter sequences work unchanged.
        att = att.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        # Softmax over the key axis: each query row becomes a distribution.
        att = F.softmax(att, dim=-1)
        attn_weights = att if return_attn else None
        att = self.attn_dropout(att)

        # Weighted sum of values: (B, h, T, T) @ (B, h, T, dk) -> (B, h, T, dk).
        y = att @ v

        # Concatenate the heads back into the model dimension:
        #   (B, h, T, dk) -> (B, T, h, dk) -> (B, T, d)
        # contiguous() is required because transpose only permutes strides, and
        # view() needs a contiguous buffer.
        y = y.transpose(1, 2).contiguous().view(B, T, d)

        y = self.resid_dropout(self.W_o(y))
        return y, attn_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network: expand, non-linearity, project back.

    Applied identically and independently at every position, so it mixes
    *features* while attention mixes *positions*. The inner dimension is wider
    than the model dimension (``4x`` by default), which is where most of a
    Transformer's parameters and most of its capacity to store learned
    transformations live. GELU is used rather than ReLU: it is smooth near
    zero, which gives better-behaved gradients for small models.
    """

    def __init__(self, config: GPTConfig) -> None:
        """Build the two linear layers and the output dropout.

        Args:
            config: Architecture configuration.
        """
        super().__init__()
        d_ff = config.d_ff or 4 * config.n_embd
        self.fc_in = nn.Linear(config.n_embd, d_ff, bias=config.bias)
        self.fc_out = nn.Linear(d_ff, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward transformation.

        Args:
            x: Tensor of shape ``(B, T, d)``.

        Returns:
            Tensor of shape ``(B, T, d)``.
        """
        return self.dropout(self.fc_out(F.gelu(self.fc_in(x))))


class TransformerBlock(nn.Module):
    """One pre-norm Transformer block: masked self-attention, then feed-forward.

    Both sublayers are wrapped in residual connections:

        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    **Pre-norm** (normalise the sublayer *input*) is used rather than post-norm.
    In a pre-norm stack the residual path from the embeddings to the output head
    is a clean sum with no normalisation in the way, so gradients reach early
    layers undistorted and deep stacks train without a long learning-rate warmup
    or careful initialisation. Post-norm places a LayerNorm on that path, which
    is what makes deep post-norm stacks notoriously sensitive to warmup.

    The residual connections are what make depth usable at all: without them
    each block would have to relearn an identity mapping, and gradients would
    have to survive multiplication through every intervening layer.
    """

    def __init__(self, config: GPTConfig) -> None:
        """Build the two normalised sublayers.

        Args:
            config: Architecture configuration.
        """
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the block.

        Args:
            x: Tensor of shape ``(B, T, d)``.
            return_attn: Whether to surface this block's attention weights.

        Returns:
            A ``(output, attn)`` pair, where ``attn`` is ``(B, h, T, T)`` or
            ``None``.
        """
        attn_out, attn_weights = self.attn(self.ln_1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights


def sinusoidal_positional_encoding(block_size: int, n_embd: int) -> torch.Tensor:
    """Build the fixed sinusoidal position table from Vaswani et al. (2017).

    Even feature indices carry sines and odd indices cosines, with wavelengths
    in geometric progression from ``2*pi`` to ``10000 * 2*pi``. Because the
    encoding is a deterministic function of the index, it extends to positions
    never seen in training — unlike learned absolute embeddings.

    Args:
        block_size: Number of positions to generate.
        n_embd: Model dimension.

    Returns:
        Tensor of shape ``(block_size, n_embd)``.
    """
    position = torch.arange(block_size, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, n_embd, 2, dtype=torch.float32) * (-math.log(10000.0) / n_embd)
    )
    pe = torch.zeros(block_size, n_embd)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
    return pe


class GPT(nn.Module):
    """A decoder-only Transformer language model.

    Forward pass, in the order the spec lays out:

    1. **Input representation.** Token ids index an embedding matrix of shape
       ``(V, d)``. Position is injected separately, because self-attention is
       permutation invariant — the attention computation treats its input as a
       *set*, so without a position signal "A is bigger than B" and "B is bigger
       than A" would produce identical representations. The two are added and
       passed through embedding dropout.

       ``pos_encoding="learned"`` allocates a ``(block_size, d)`` table, which
       makes ``block_size`` a hard ceiling on sequence length: position 600 has
       no row in a 512-row table. ``"sinusoidal"`` is a closed-form function of
       the index and so extrapolates in principle. ``"none"`` exists for the
       optional bonus ablation.

    2. **Blocks.** ``n_layer`` pre-norm blocks of masked multi-head attention
       plus a feed-forward network, each residually connected.

    3. **Final norm and output head.** A final LayerNorm, then a linear map from
       ``d`` to ``V`` producing next-token logits. When ``tie_weights`` is set,
       this reuses the input embedding matrix instead of allocating its own,
       saving ``V * d`` parameters (4.1M for Hindi's 8,000-token vocabulary,
       5.1M for Nepali's 10,000) — a large fraction of a ~25M model — and
       forcing the input and output representations of a token to coincide.

    4. **Objective.** Cross-entropy between the logits at position ``t`` and the
       token at position ``t+1``, averaged over all positions.
    """

    def __init__(self, config: GPTConfig) -> None:
        """Build the full model and initialise its weights.

        Args:
            config: Architecture configuration.
        """
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        if config.pos_encoding == "learned":
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        elif config.pos_encoding == "sinusoidal":
            self.position_embedding = None
            self.register_buffer(
                "pos_table",
                sinusoidal_positional_encoding(config.block_size, config.n_embd),
                persistent=False,
            )
        else:  # "none" — the bonus ablation
            self.position_embedding = None

        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            # Weight tying: the same (V, d) matrix serves as the input lookup
            # table and the output projection. nn.Linear stores its weight as
            # (out_features, in_features) = (V, d), which matches nn.Embedding's
            # (V, d) exactly, so the two can share storage directly.
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # Scaled initialisation for the projections that write into the residual
        # stream. With n_layer blocks each adding to the stream, the variance of
        # the residual grows linearly with depth; shrinking these by
        # 1/sqrt(2 * n_layer) keeps the stream at roughly unit scale at init.
        for name, param in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=config.init_std / math.sqrt(2 * config.n_layer),
                )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise linear and embedding weights from a small normal.

        Args:
            module: The submodule being visited by ``nn.Module.apply``.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def parameter_count(self) -> dict[str, int]:
        """Count parameters, both in total and excluding the embedding tables.

        The non-embedding count is the fairer number when comparing Model H and
        Model L, since their vocabularies differ (8,000 vs. 10,000) and so their
        embedding tables differ in size by construction.

        Returns:
            Mapping with ``total``, ``trainable``, ``embedding`` and
            ``non_embedding`` counts. Tied weights are counted once.
        """
        seen: set[int] = set()
        total = 0
        trainable = 0
        for param in self.parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        embedding = self.token_embedding.weight.numel()
        if self.position_embedding is not None:
            embedding += self.position_embedding.weight.numel()
        return {
            "total": total,
            "trainable": trainable,
            "embedding": embedding,
            "non_embedding": total - embedding,
        }

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        """Run the model over a batch of token ids.

        Args:
            idx: Token ids of shape ``(B, T)``.
            targets: Optional next-token targets of shape ``(B, T)``. When
                given, the causal language-modelling loss is returned.
            return_attn: Whether to collect per-layer attention probabilities.

        Returns:
            A ``(logits, loss, attentions)`` triple. ``logits`` is ``(B, T, V)``,
            ``loss`` is a scalar or ``None``, and ``attentions`` is a list of
            ``n_layer`` tensors of shape ``(B, h, T, T)`` or ``None``.

        Raises:
            ValueError: If ``T`` exceeds ``block_size``.
        """
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.config.block_size}"
            )

        # (B, T) -> (B, T, d)
        x = self.token_embedding(idx)

        if self.config.pos_encoding == "learned":
            pos = torch.arange(T, device=idx.device)
            x = x + self.position_embedding(pos)  # (T, d) broadcasts over batch
        elif self.config.pos_encoding == "sinusoidal":
            x = x + self.pos_table[:T].to(x.dtype)
        # "none": position information is deliberately withheld (bonus ablation)

        x = self.embedding_dropout(x)

        attentions: list[torch.Tensor] | None = [] if return_attn else None
        for block in self.blocks:
            x, attn = block(x, return_attn=return_attn)
            if attentions is not None and attn is not None:
                attentions.append(attn)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        if targets is not None:
            # Flatten batch and time so cross_entropy sees (B*T, V) vs (B*T,).
            # The caller supplies targets already shifted by one position.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss, attentions

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        greedy: bool = False,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend a prompt.

        Each step feeds the current sequence back in and samples from the
        distribution at the final position. The context is cropped to the last
        ``block_size`` tokens, because positions beyond that have no positional
        embedding.

        Args:
            idx: Prompt token ids of shape ``(B, T)``.
            max_new_tokens: Number of tokens to append.
            temperature: Softmax temperature. Values below 1 sharpen the
                distribution toward the mode; values above 1 flatten it,
                trading fluency for diversity.
            top_k: If set, restrict sampling to the ``k`` highest-probability
                tokens at each step.
            greedy: If True, take the argmax instead of sampling, ignoring
                ``temperature`` and ``top_k``.
            eos_id: If set, stop early once every sequence has emitted EOS.

        Returns:
            Token ids of shape ``(B, T + n)``, where ``n <= max_new_tokens``.
        """
        was_training = self.training
        self.eval()
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, V): only the last position matters
            if greedy:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)[0]
                    logits = logits.masked_fill(logits < kth[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
            if eos_id is not None:
                finished |= next_token.squeeze(1) == eos_id
                if bool(finished.all()):
                    break
        if was_training:
            self.train()
        return idx

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str = "cuda",
    ) -> torch.optim.Optimizer:
        """Build AdamW with weight decay applied only to matrix parameters.

        Decaying biases, LayerNorm gains and embeddings is standard practice to
        avoid: those parameters set scales and offsets rather than performing
        the linear mixing that decay is meant to regularise, and shrinking them
        toward zero hurts more than it helps.

        Args:
            weight_decay: Decay coefficient for the decayed group.
            learning_rate: Peak learning rate.
            betas: AdamW ``(beta1, beta2)``.
            device_type: ``"cuda"`` or ``"cpu"``; enables the fused kernel on CUDA.

        Returns:
            The configured optimizer.
        """
        decay, no_decay, seen = [], [], set()
        for _, param in self.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            (decay if param.dim() >= 2 else no_decay).append(param)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            groups,
            lr=learning_rate,
            betas=betas,
            fused=(device_type == "cuda"),
        )
