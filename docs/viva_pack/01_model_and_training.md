# Phase 2 - the Transformer and the training loop

Part of the LMA individual project viva pack. Line numbers below match the files in the repository exactly, so any line cited here can be opened directly in the IDE.

Files in this pack:

- `hindi/model/transformer.py`
- `hindi/model/config.py`
- `hindi/train/train.py`
- `hindi/configs/model.yaml`
- `nepali/configs/model.yaml`


## hindi/model/transformer.py

```python
   1  """A decoder-only (GPT-style) Transformer language model, written from scratch.
   2  
   3  Everything in this file is built from primitive PyTorch layers only —
   4  ``nn.Linear``, ``nn.Embedding``, ``nn.Dropout``, ``nn.Parameter`` and plain
   5  tensor ops. No ``nn.Transformer*`` module, no ``nn.MultiheadAttention``, no
   6  ``F.scaled_dot_product_attention``, and no HuggingFace model class is used
   7  anywhere. The attention scores, the causal mask and the softmax are computed
   8  explicitly so that every tensor operation in the forward pass is visible, and
   9  so that the attention probability matrices can be returned for the Phase 2/3
  10  attention analysis.
  11  
  12  Shape conventions used throughout:
  13  
  14      B  batch size
  15      T  sequence length (number of tokens), ``T <= config.block_size``
  16      d  model dimension (``config.n_embd``)
  17      h  number of attention heads (``config.n_head``)
  18      dk per-head dimension (``d / h``)
  19      V  vocabulary size
  20  """
  21  
  22  from __future__ import annotations
  23  
  24  import math
  25  
  26  import torch
  27  import torch.nn as nn
  28  import torch.nn.functional as F
  29  
  30  from .config import GPTConfig
  31  
  32  
  33  class LayerNorm(nn.Module):
  34      """Layer normalisation with an optional bias term.
  35  
  36      ``nn.LayerNorm`` always allocates a bias when affine; this thin wrapper
  37      around ``F.layer_norm`` lets the bias be dropped, which is a common choice
  38      in modern GPT implementations and removes a small number of parameters that
  39      empirically do little.
  40  
  41      Normalisation is over the feature dimension only — each token position is
  42      normalised independently of the others, which is what keeps the operation
  43      compatible with causal masking (it never mixes information across time).
  44      """
  45  
  46      def __init__(self, ndim: int, bias: bool = True) -> None:
  47          """Create a LayerNorm over the last dimension.
  48  
  49          Args:
  50              ndim: Size of the normalised (feature) dimension.
  51              bias: Whether to learn an additive shift alongside the gain.
  52          """
  53          super().__init__()
  54          self.weight = nn.Parameter(torch.ones(ndim))
  55          self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
  56  
  57      def forward(self, x: torch.Tensor) -> torch.Tensor:
  58          """Normalise the final dimension of ``x``.
  59  
  60          Args:
  61              x: Tensor of shape ``(..., ndim)``.
  62  
  63          Returns:
  64              Tensor of the same shape, normalised then scaled and shifted.
  65          """
  66          return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
  67  
  68  
  69  class MultiHeadCausalSelfAttention(nn.Module):
  70      """Multi-head causal self-attention, implemented from first principles.
  71  
  72      For an input ``X`` of shape ``(B, T, d)`` this computes, per head:
  73  
  74          Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(dk) + M ) V
  75  
  76      where ``Q = X W_Q``, ``K = X W_K``, ``V = X W_V``, and ``M`` is an additive
  77      causal mask holding ``0`` on allowed positions and ``-inf`` on future ones.
  78  
  79      Why the ``1/sqrt(dk)`` scaling: ``Q`` and ``K`` entries are roughly unit
  80      variance at initialisation, so a dot product over ``dk`` dimensions has
  81      variance on the order of ``dk``. Without rescaling, the logits fed to the
  82      softmax grow with ``dk``; the softmax then saturates, producing a
  83      near-one-hot distribution whose gradients vanish. Dividing by ``sqrt(dk)``
  84      restores unit-scale logits and keeps the softmax in its responsive regime.
  85  
  86      Why the mask: this is a *causal* language model, so the prediction at
  87      position ``t`` may depend only on positions ``<= t``. Setting future scores
  88      to ``-inf`` before the softmax drives their probabilities to exactly zero,
  89      so no information flows backwards in time. Removing the mask would let the
  90      model read the answer it is being trained to predict, and validation loss
  91      would collapse toward zero while the model learned nothing useful.
  92  
  93      The three projections are kept as separate ``nn.Linear`` modules rather than
  94      fused into one ``(d, 3d)`` matmul, so that ``W_Q``, ``W_K`` and ``W_V`` map
  95      one-to-one onto the mathematical description above.
  96      """
  97  
  98      def __init__(self, config: GPTConfig) -> None:
  99          """Build the projections, dropouts and causal mask buffer.
 100  
 101          Args:
 102              config: Architecture configuration.
 103          """
 104          super().__init__()
 105          self.n_head = config.n_head
 106          self.n_embd = config.n_embd
 107          self.head_dim = config.head_dim
 108  
 109          # Q, K, V projections: (B, T, d) -> (B, T, d) each.
 110          self.W_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
 111          self.W_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
 112          self.W_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
 113          # Output projection applied after the heads are concatenated back to d.
 114          self.W_o = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
 115  
 116          self.attn_dropout = nn.Dropout(config.dropout)
 117          self.resid_dropout = nn.Dropout(config.dropout)
 118  
 119          # Upper-triangular boolean mask, True where a position is in the future
 120          # and must be blocked. Registered as a non-persistent buffer: it moves
 121          # with the module across devices but is not written into checkpoints,
 122          # since it is fully determined by block_size.
 123          causal = torch.triu(
 124              torch.ones(config.block_size, config.block_size, dtype=torch.bool),
 125              diagonal=1,
 126          )
 127          self.register_buffer("causal_mask", causal, persistent=False)
 128  
 129      def forward(
 130          self, x: torch.Tensor, return_attn: bool = False
 131      ) -> tuple[torch.Tensor, torch.Tensor | None]:
 132          """Run masked multi-head self-attention over a sequence.
 133  
 134          Args:
 135              x: Input activations of shape ``(B, T, d)``.
 136              return_attn: If True, also return the post-softmax attention
 137                  probabilities. Kept optional because materialising a
 138                  ``(B, h, T, T)`` tensor is expensive and only the analysis code
 139                  needs it.
 140  
 141          Returns:
 142              A ``(output, attn)`` pair. ``output`` has shape ``(B, T, d)``.
 143              ``attn`` is ``(B, h, T, T)`` when requested, else ``None``.
 144  
 145          Raises:
 146              ValueError: If ``T`` exceeds the configured ``block_size``, since
 147                  the causal mask buffer is only defined up to that length.
 148          """
 149          B, T, d = x.shape
 150          if T > self.causal_mask.size(0):
 151              raise ValueError(
 152                  f"sequence length {T} exceeds block_size {self.causal_mask.size(0)}"
 153              )
 154  
 155          # (B, T, d) -> (B, T, d), one projection per role.
 156          q = self.W_q(x)
 157          k = self.W_k(x)
 158          v = self.W_v(x)
 159  
 160          # Split the model dimension into h heads of width dk, then move the head
 161          # axis in front of time so each head is an independent (T, dk) sequence:
 162          #   (B, T, d) -> (B, T, h, dk) -> (B, h, T, dk)
 163          q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
 164          k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
 165          v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
 166  
 167          # Scaled dot-product scores: (B, h, T, dk) @ (B, h, dk, T) -> (B, h, T, T).
 168          # Entry [b, head, i, j] is how much query position i attends to key j.
 169          att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
 170  
 171          # Additive causal mask: -inf on future positions, so softmax sends them
 172          # to exactly zero. Sliced to T so shorter sequences work unchanged.
 173          att = att.masked_fill(self.causal_mask[:T, :T], float("-inf"))
 174  
 175          # Softmax over the key axis: each query row becomes a distribution.
 176          att = F.softmax(att, dim=-1)
 177          attn_weights = att if return_attn else None
 178          att = self.attn_dropout(att)
 179  
 180          # Weighted sum of values: (B, h, T, T) @ (B, h, T, dk) -> (B, h, T, dk).
 181          y = att @ v
 182  
 183          # Concatenate the heads back into the model dimension:
 184          #   (B, h, T, dk) -> (B, T, h, dk) -> (B, T, d)
 185          # contiguous() is required because transpose only permutes strides, and
 186          # view() needs a contiguous buffer.
 187          y = y.transpose(1, 2).contiguous().view(B, T, d)
 188  
 189          y = self.resid_dropout(self.W_o(y))
 190          return y, attn_weights
 191  
 192  
 193  class FeedForward(nn.Module):
 194      """Position-wise feed-forward network: expand, non-linearity, project back.
 195  
 196      Applied identically and independently at every position, so it mixes
 197      *features* while attention mixes *positions*. The inner dimension is wider
 198      than the model dimension (``4x`` by default), which is where most of a
 199      Transformer's parameters and most of its capacity to store learned
 200      transformations live. GELU is used rather than ReLU: it is smooth near
 201      zero, which gives better-behaved gradients for small models.
 202      """
 203  
 204      def __init__(self, config: GPTConfig) -> None:
 205          """Build the two linear layers and the output dropout.
 206  
 207          Args:
 208              config: Architecture configuration.
 209          """
 210          super().__init__()
 211          d_ff = config.d_ff or 4 * config.n_embd
 212          self.fc_in = nn.Linear(config.n_embd, d_ff, bias=config.bias)
 213          self.fc_out = nn.Linear(d_ff, config.n_embd, bias=config.bias)
 214          self.dropout = nn.Dropout(config.dropout)
 215  
 216      def forward(self, x: torch.Tensor) -> torch.Tensor:
 217          """Apply the feed-forward transformation.
 218  
 219          Args:
 220              x: Tensor of shape ``(B, T, d)``.
 221  
 222          Returns:
 223              Tensor of shape ``(B, T, d)``.
 224          """
 225          return self.dropout(self.fc_out(F.gelu(self.fc_in(x))))
 226  
 227  
 228  class TransformerBlock(nn.Module):
 229      """One pre-norm Transformer block: masked self-attention, then feed-forward.
 230  
 231      Both sublayers are wrapped in residual connections:
 232  
 233          x = x + Attention(LayerNorm(x))
 234          x = x + FeedForward(LayerNorm(x))
 235  
 236      **Pre-norm** (normalise the sublayer *input*) is used rather than post-norm.
 237      In a pre-norm stack the residual path from the embeddings to the output head
 238      is a clean sum with no normalisation in the way, so gradients reach early
 239      layers undistorted and deep stacks train without a long learning-rate warmup
 240      or careful initialisation. Post-norm places a LayerNorm on that path, which
 241      is what makes deep post-norm stacks notoriously sensitive to warmup.
 242  
 243      The residual connections are what make depth usable at all: without them
 244      each block would have to relearn an identity mapping, and gradients would
 245      have to survive multiplication through every intervening layer.
 246      """
 247  
 248      def __init__(self, config: GPTConfig) -> None:
 249          """Build the two normalised sublayers.
 250  
 251          Args:
 252              config: Architecture configuration.
 253          """
 254          super().__init__()
 255          self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
 256          self.attn = MultiHeadCausalSelfAttention(config)
 257          self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
 258          self.mlp = FeedForward(config)
 259  
 260      def forward(
 261          self, x: torch.Tensor, return_attn: bool = False
 262      ) -> tuple[torch.Tensor, torch.Tensor | None]:
 263          """Run the block.
 264  
 265          Args:
 266              x: Tensor of shape ``(B, T, d)``.
 267              return_attn: Whether to surface this block's attention weights.
 268  
 269          Returns:
 270              A ``(output, attn)`` pair, where ``attn`` is ``(B, h, T, T)`` or
 271              ``None``.
 272          """
 273          attn_out, attn_weights = self.attn(self.ln_1(x), return_attn=return_attn)
 274          x = x + attn_out
 275          x = x + self.mlp(self.ln_2(x))
 276          return x, attn_weights
 277  
 278  
 279  def sinusoidal_positional_encoding(block_size: int, n_embd: int) -> torch.Tensor:
 280      """Build the fixed sinusoidal position table from Vaswani et al. (2017).
 281  
 282      Even feature indices carry sines and odd indices cosines, with wavelengths
 283      in geometric progression from ``2*pi`` to ``10000 * 2*pi``. Because the
 284      encoding is a deterministic function of the index, it extends to positions
 285      never seen in training — unlike learned absolute embeddings.
 286  
 287      Args:
 288          block_size: Number of positions to generate.
 289          n_embd: Model dimension.
 290  
 291      Returns:
 292          Tensor of shape ``(block_size, n_embd)``.
 293      """
 294      position = torch.arange(block_size, dtype=torch.float32).unsqueeze(1)
 295      div_term = torch.exp(
 296          torch.arange(0, n_embd, 2, dtype=torch.float32) * (-math.log(10000.0) / n_embd)
 297      )
 298      pe = torch.zeros(block_size, n_embd)
 299      pe[:, 0::2] = torch.sin(position * div_term)
 300      pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
 301      return pe
 302  
 303  
 304  class GPT(nn.Module):
 305      """A decoder-only Transformer language model.
 306  
 307      Forward pass, in the order the spec lays out:
 308  
 309      1. **Input representation.** Token ids index an embedding matrix of shape
 310         ``(V, d)``. Position is injected separately, because self-attention is
 311         permutation invariant — the attention computation treats its input as a
 312         *set*, so without a position signal "A is bigger than B" and "B is bigger
 313         than A" would produce identical representations. The two are added and
 314         passed through embedding dropout.
 315  
 316         ``pos_encoding="learned"`` allocates a ``(block_size, d)`` table, which
 317         makes ``block_size`` a hard ceiling on sequence length: position 600 has
 318         no row in a 512-row table. ``"sinusoidal"`` is a closed-form function of
 319         the index and so extrapolates in principle. ``"none"`` exists for the
 320         optional bonus ablation.
 321  
 322      2. **Blocks.** ``n_layer`` pre-norm blocks of masked multi-head attention
 323         plus a feed-forward network, each residually connected.
 324  
 325      3. **Final norm and output head.** A final LayerNorm, then a linear map from
 326         ``d`` to ``V`` producing next-token logits. When ``tie_weights`` is set,
 327         this reuses the input embedding matrix instead of allocating its own,
 328         saving ``V * d`` parameters (4.1M for Hindi's 8,000-token vocabulary,
 329         5.1M for Nepali's 10,000) — a large fraction of a ~25M model — and
 330         forcing the input and output representations of a token to coincide.
 331  
 332      4. **Objective.** Cross-entropy between the logits at position ``t`` and the
 333         token at position ``t+1``, averaged over all positions.
 334      """
 335  
 336      def __init__(self, config: GPTConfig) -> None:
 337          """Build the full model and initialise its weights.
 338  
 339          Args:
 340              config: Architecture configuration.
 341          """
 342          super().__init__()
 343          self.config = config
 344  
 345          self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
 346          if config.pos_encoding == "learned":
 347              self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
 348          elif config.pos_encoding == "sinusoidal":
 349              self.position_embedding = None
 350              self.register_buffer(
 351                  "pos_table",
 352                  sinusoidal_positional_encoding(config.block_size, config.n_embd),
 353                  persistent=False,
 354              )
 355          else:  # "none" — the bonus ablation
 356              self.position_embedding = None
 357  
 358          self.embedding_dropout = nn.Dropout(config.dropout)
 359          self.blocks = nn.ModuleList(
 360              [TransformerBlock(config) for _ in range(config.n_layer)]
 361          )
 362          self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
 363          self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
 364  
 365          if config.tie_weights:
 366              # Weight tying: the same (V, d) matrix serves as the input lookup
 367              # table and the output projection. nn.Linear stores its weight as
 368              # (out_features, in_features) = (V, d), which matches nn.Embedding's
 369              # (V, d) exactly, so the two can share storage directly.
 370              self.lm_head.weight = self.token_embedding.weight
 371  
 372          self.apply(self._init_weights)
 373          # Scaled initialisation for the projections that write into the residual
 374          # stream. With n_layer blocks each adding to the stream, the variance of
 375          # the residual grows linearly with depth; shrinking these by
 376          # 1/sqrt(2 * n_layer) keeps the stream at roughly unit scale at init.
 377          for name, param in self.named_parameters():
 378              if name.endswith("W_o.weight") or name.endswith("fc_out.weight"):
 379                  nn.init.normal_(
 380                      param,
 381                      mean=0.0,
 382                      std=config.init_std / math.sqrt(2 * config.n_layer),
 383                  )
 384  
 385      def _init_weights(self, module: nn.Module) -> None:
 386          """Initialise linear and embedding weights from a small normal.
 387  
 388          Args:
 389              module: The submodule being visited by ``nn.Module.apply``.
 390          """
 391          if isinstance(module, nn.Linear):
 392              nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
 393              if module.bias is not None:
 394                  nn.init.zeros_(module.bias)
 395          elif isinstance(module, nn.Embedding):
 396              nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
 397  
 398      def parameter_count(self) -> dict[str, int]:
 399          """Count parameters, both in total and excluding the embedding tables.
 400  
 401          The non-embedding count is the fairer number when comparing Model H and
 402          Model L, since their vocabularies differ (8,000 vs. 10,000) and so their
 403          embedding tables differ in size by construction.
 404  
 405          Returns:
 406              Mapping with ``total``, ``trainable``, ``embedding`` and
 407              ``non_embedding`` counts. Tied weights are counted once.
 408          """
 409          seen: set[int] = set()
 410          total = 0
 411          trainable = 0
 412          for param in self.parameters():
 413              if id(param) in seen:
 414                  continue
 415              seen.add(id(param))
 416              total += param.numel()
 417              if param.requires_grad:
 418                  trainable += param.numel()
 419          embedding = self.token_embedding.weight.numel()
 420          if self.position_embedding is not None:
 421              embedding += self.position_embedding.weight.numel()
 422          return {
 423              "total": total,
 424              "trainable": trainable,
 425              "embedding": embedding,
 426              "non_embedding": total - embedding,
 427          }
 428  
 429      def forward(
 430          self,
 431          idx: torch.Tensor,
 432          targets: torch.Tensor | None = None,
 433          return_attn: bool = False,
 434      ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
 435          """Run the model over a batch of token ids.
 436  
 437          Args:
 438              idx: Token ids of shape ``(B, T)``.
 439              targets: Optional next-token targets of shape ``(B, T)``. When
 440                  given, the causal language-modelling loss is returned.
 441              return_attn: Whether to collect per-layer attention probabilities.
 442  
 443          Returns:
 444              A ``(logits, loss, attentions)`` triple. ``logits`` is ``(B, T, V)``,
 445              ``loss`` is a scalar or ``None``, and ``attentions`` is a list of
 446              ``n_layer`` tensors of shape ``(B, h, T, T)`` or ``None``.
 447  
 448          Raises:
 449              ValueError: If ``T`` exceeds ``block_size``.
 450          """
 451          B, T = idx.shape
 452          if T > self.config.block_size:
 453              raise ValueError(
 454                  f"sequence length {T} exceeds block_size {self.config.block_size}"
 455              )
 456  
 457          # (B, T) -> (B, T, d)
 458          x = self.token_embedding(idx)
 459  
 460          if self.config.pos_encoding == "learned":
 461              pos = torch.arange(T, device=idx.device)
 462              x = x + self.position_embedding(pos)  # (T, d) broadcasts over batch
 463          elif self.config.pos_encoding == "sinusoidal":
 464              x = x + self.pos_table[:T].to(x.dtype)
 465          # "none": position information is deliberately withheld (bonus ablation)
 466  
 467          x = self.embedding_dropout(x)
 468  
 469          attentions: list[torch.Tensor] | None = [] if return_attn else None
 470          for block in self.blocks:
 471              x, attn = block(x, return_attn=return_attn)
 472              if attentions is not None and attn is not None:
 473                  attentions.append(attn)
 474  
 475          x = self.ln_f(x)
 476          logits = self.lm_head(x)  # (B, T, V)
 477  
 478          loss = None
 479          if targets is not None:
 480              # Flatten batch and time so cross_entropy sees (B*T, V) vs (B*T,).
 481              # The caller supplies targets already shifted by one position.
 482              loss = F.cross_entropy(
 483                  logits.reshape(-1, logits.size(-1)),
 484                  targets.reshape(-1),
 485                  ignore_index=-1,
 486              )
 487          return logits, loss, attentions
 488  
 489      @torch.no_grad()
 490      def generate(
 491          self,
 492          idx: torch.Tensor,
 493          max_new_tokens: int,
 494          temperature: float = 1.0,
 495          top_k: int | None = None,
 496          greedy: bool = False,
 497          eos_id: int | None = None,
 498      ) -> torch.Tensor:
 499          """Autoregressively extend a prompt.
 500  
 501          Each step feeds the current sequence back in and samples from the
 502          distribution at the final position. The context is cropped to the last
 503          ``block_size`` tokens, because positions beyond that have no positional
 504          embedding.
 505  
 506          Args:
 507              idx: Prompt token ids of shape ``(B, T)``.
 508              max_new_tokens: Number of tokens to append.
 509              temperature: Softmax temperature. Values below 1 sharpen the
 510                  distribution toward the mode; values above 1 flatten it,
 511                  trading fluency for diversity.
 512              top_k: If set, restrict sampling to the ``k`` highest-probability
 513                  tokens at each step.
 514              greedy: If True, take the argmax instead of sampling, ignoring
 515                  ``temperature`` and ``top_k``.
 516              eos_id: If set, stop early once every sequence has emitted EOS.
 517  
 518          Returns:
 519              Token ids of shape ``(B, T + n)``, where ``n <= max_new_tokens``.
 520          """
 521          was_training = self.training
 522          self.eval()
 523          finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
 524          for _ in range(max_new_tokens):
 525              idx_cond = idx[:, -self.config.block_size :]
 526              logits, _, _ = self(idx_cond)
 527              logits = logits[:, -1, :]  # (B, V): only the last position matters
 528              if greedy:
 529                  next_token = logits.argmax(dim=-1, keepdim=True)
 530              else:
 531                  logits = logits / max(temperature, 1e-6)
 532                  if top_k is not None:
 533                      kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)[0]
 534                      logits = logits.masked_fill(logits < kth[:, [-1]], float("-inf"))
 535                  probs = F.softmax(logits, dim=-1)
 536                  next_token = torch.multinomial(probs, num_samples=1)
 537              idx = torch.cat([idx, next_token], dim=1)
 538              if eos_id is not None:
 539                  finished |= next_token.squeeze(1) == eos_id
 540                  if bool(finished.all()):
 541                      break
 542          if was_training:
 543              self.train()
 544          return idx
 545  
 546      def configure_optimizers(
 547          self,
 548          weight_decay: float,
 549          learning_rate: float,
 550          betas: tuple[float, float],
 551          device_type: str = "cuda",
 552      ) -> torch.optim.Optimizer:
 553          """Build AdamW with weight decay applied only to matrix parameters.
 554  
 555          Decaying biases, LayerNorm gains and embeddings is standard practice to
 556          avoid: those parameters set scales and offsets rather than performing
 557          the linear mixing that decay is meant to regularise, and shrinking them
 558          toward zero hurts more than it helps.
 559  
 560          Args:
 561              weight_decay: Decay coefficient for the decayed group.
 562              learning_rate: Peak learning rate.
 563              betas: AdamW ``(beta1, beta2)``.
 564              device_type: ``"cuda"`` or ``"cpu"``; enables the fused kernel on CUDA.
 565  
 566          Returns:
 567              The configured optimizer.
 568          """
 569          decay, no_decay, seen = [], [], set()
 570          for _, param in self.named_parameters():
 571              if not param.requires_grad or id(param) in seen:
 572                  continue
 573              seen.add(id(param))
 574              (decay if param.dim() >= 2 else no_decay).append(param)
 575          groups = [
 576              {"params": decay, "weight_decay": weight_decay},
 577              {"params": no_decay, "weight_decay": 0.0},
 578          ]
 579          return torch.optim.AdamW(
 580              groups,
 581              lr=learning_rate,
 582              betas=betas,
 583              fused=(device_type == "cuda"),
 584          )
```


## hindi/model/config.py

```python
   1  """Model and training configuration objects, loadable from and dumpable to YAML.
   2  
   3  Every run's full configuration is written next to its checkpoints, so a run can
   4  be reproduced (and resumed) from the repository alone, as the spec requires.
   5  The two languages use *separate* config files with different vocabulary sizes;
   6  nothing here is shared at runtime between Model H and Model L.
   7  """
   8  
   9  from __future__ import annotations
  10  
  11  import json
  12  from dataclasses import asdict, dataclass, field, fields
  13  from pathlib import Path
  14  from typing import Any
  15  
  16  import yaml
  17  
  18  # Positional-encoding schemes this implementation supports. "none" exists for
  19  # the optional Phase 2 bonus ablation (retrain with position information
  20  # removed) — it is a config flag rather than a separate model class so the
  21  # ablation runs through exactly the same code path as the standard model.
  22  POS_ENCODINGS = ("learned", "sinusoidal", "none")
  23  
  24  
  25  @dataclass
  26  class GPTConfig:
  27      """Architecture hyperparameters for one decoder-only Transformer.
  28  
  29      The defaults target the spec's ~25M trainable parameters. With
  30      ``n_embd=512`` and ``n_layer=6`` the block stack contributes roughly
  31      ``12 * n_embd^2`` parameters per layer (``4 * n_embd^2`` for the four
  32      attention projections, ``8 * n_embd^2`` for the two feed-forward layers at
  33      ``d_ff = 4 * n_embd``), giving ~18.9M, on top of a ``vocab_size * n_embd``
  34      embedding table that differs between the two languages because their
  35      vocabularies differ.
  36  
  37      Attributes:
  38          vocab_size: Size of this language's tokenizer vocabulary. Differs
  39              between Model H (8,000) and Model L (10,000); the embedding and
  40              output layers therefore differ in size between the two models.
  41          block_size: Maximum context length in tokens. With learned absolute
  42              positional embeddings this is a hard ceiling: the model has no
  43              embedding row for a position beyond ``block_size``, so it cannot be
  44              evaluated on longer sequences without interpolating or retraining
  45              the positional table.
  46          n_layer: Number of stacked Transformer blocks.
  47          n_head: Number of attention heads. Must divide ``n_embd``.
  48          n_embd: Model dimension ``d``.
  49          d_ff: Inner width of the position-wise feed-forward network. Defaults to
  50              ``4 * n_embd`` when left as ``None``.
  51          dropout: Dropout probability used for embeddings, attention
  52              probabilities and sublayer outputs.
  53          bias: Whether linear layers and LayerNorms carry bias terms.
  54          tie_weights: Whether the output projection reuses the input embedding
  55              matrix. Saves ``vocab_size * n_embd`` parameters (4.1M for Hindi,
  56              5.1M for Nepali) and ties the two representations of a token.
  57          pos_encoding: One of :data:`POS_ENCODINGS`.
  58          init_std: Standard deviation for normal weight initialisation.
  59      """
  60  
  61      vocab_size: int
  62      block_size: int = 512
  63      n_layer: int = 6
  64      n_head: int = 8
  65      n_embd: int = 512
  66      d_ff: int | None = None
  67      dropout: float = 0.1
  68      bias: bool = True
  69      tie_weights: bool = True
  70      pos_encoding: str = "learned"
  71      init_std: float = 0.02
  72  
  73      def __post_init__(self) -> None:
  74          """Fill in derived defaults and reject invalid combinations early.
  75  
  76          Raises:
  77              ValueError: If the head count does not divide the model dimension,
  78                  or an unknown positional-encoding scheme is requested.
  79          """
  80          if self.d_ff is None:
  81              self.d_ff = 4 * self.n_embd
  82          if self.n_embd % self.n_head != 0:
  83              raise ValueError(
  84                  f"n_embd={self.n_embd} must be divisible by n_head={self.n_head}; "
  85                  "every head needs an equal-width d_k = n_embd / n_head subspace"
  86              )
  87          if self.pos_encoding not in POS_ENCODINGS:
  88              raise ValueError(
  89                  f"pos_encoding={self.pos_encoding!r} not in {POS_ENCODINGS}"
  90              )
  91  
  92      @property
  93      def head_dim(self) -> int:
  94          """Width ``d_k`` of each attention head's subspace."""
  95          return self.n_embd // self.n_head
  96  
  97      def estimated_parameters(self) -> dict[str, int]:
  98          """Analytically estimate the parameter count, broken down by component.
  99  
 100          This is a paper estimate used for sizing configs before instantiating a
 101          model; :meth:`~hindi.model.transformer.GPT.parameter_count` reports the true count
 102          from the built module. The two should agree to within the bias terms.
 103  
 104          Returns:
 105              Mapping from component name to parameter count, including a
 106              ``"total"`` key.
 107          """
 108          d, v, L = self.n_embd, self.vocab_size, self.n_layer
 109          d_ff = self.d_ff or 4 * d
 110          token_emb = v * d
 111          pos_emb = self.block_size * d if self.pos_encoding == "learned" else 0
 112          attn_per_layer = 4 * d * d + (4 * d if self.bias else 0)
 113          ffn_per_layer = 2 * d * d_ff + ((d_ff + d) if self.bias else 0)
 114          norm_per_layer = 2 * (2 * d if self.bias else d)
 115          blocks = L * (attn_per_layer + ffn_per_layer + norm_per_layer)
 116          final_norm = 2 * d if self.bias else d
 117          head = 0 if self.tie_weights else v * d
 118          total = token_emb + pos_emb + blocks + final_norm + head
 119          return {
 120              "token_embedding": token_emb,
 121              "positional_embedding": pos_emb,
 122              "blocks": blocks,
 123              "final_norm": final_norm,
 124              "lm_head": head,
 125              "total": total,
 126          }
 127  
 128      def to_dict(self) -> dict[str, Any]:
 129          """Return this config as a plain dictionary."""
 130          return asdict(self)
 131  
 132      @classmethod
 133      def from_dict(cls, data: dict[str, Any]) -> "GPTConfig":
 134          """Build a config from a dictionary, ignoring unknown keys.
 135  
 136          Args:
 137              data: Mapping of field names to values.
 138  
 139          Returns:
 140              The constructed :class:`GPTConfig`.
 141          """
 142          known = {f.name for f in fields(cls)}
 143          return cls(**{k: v for k, v in data.items() if k in known})
 144  
 145  
 146  @dataclass
 147  class TrainConfig:
 148      """Optimisation, schedule, checkpointing and data-location settings.
 149  
 150      Attributes:
 151          language: Language name, used for run naming and output paths.
 152          data_dir: Directory holding ``train.bin`` / ``val.bin`` / ``test.bin``.
 153          out_dir: Directory for checkpoints and the run's config snapshot.
 154          max_steps: Total optimiser steps.
 155          batch_size: Sequences per micro-batch (per forward pass).
 156          grad_accum_steps: Micro-batches accumulated per optimiser step. The
 157              effective batch is ``batch_size * grad_accum_steps * block_size``
 158              tokens; this is the knob that decouples the effective batch from
 159              available VRAM.
 160          learning_rate: Peak learning rate after warmup.
 161          min_lr: Floor the cosine schedule decays to.
 162          warmup_steps: Linear warmup length in steps.
 163          weight_decay: AdamW decoupled weight decay, applied only to matrices
 164              (not to biases, LayerNorm gains or embeddings).
 165          beta1: AdamW first-moment decay.
 166          beta2: AdamW second-moment decay.
 167          grad_clip: Global gradient-norm clip; ``0`` disables clipping.
 168          eval_interval: Steps between validation passes.
 169          eval_iters: Micro-batches averaged per validation pass.
 170          log_interval: Steps between training-loss log lines.
 171          checkpoint_interval: Steps between checkpoint writes.
 172          keep_last_n: How many rolling checkpoints to retain on disk.
 173          seed: Base RNG seed.
 174          device: ``"cuda"``, ``"cpu"``, or ``"auto"``.
 175          dtype: Autocast dtype: ``"bfloat16"``, ``"float16"`` or ``"float32"``.
 176          compile_model: Whether to wrap the model in ``torch.compile``.
 177      """
 178  
 179      language: str
 180      data_dir: str
 181      out_dir: str
 182      max_steps: int = 20_000
 183      batch_size: int = 16
 184      grad_accum_steps: int = 8
 185      learning_rate: float = 6e-4
 186      min_lr: float = 6e-5
 187      warmup_steps: int = 500
 188      weight_decay: float = 0.1
 189      beta1: float = 0.9
 190      beta2: float = 0.95
 191      grad_clip: float = 1.0
 192      eval_interval: int = 500
 193      eval_iters: int = 100
 194      log_interval: int = 20
 195      checkpoint_interval: int = 500
 196      keep_last_n: int = 2
 197      seed: int = 20260902
 198      device: str = "auto"
 199      dtype: str = "bfloat16"
 200      compile_model: bool = False
 201  
 202      def tokens_per_step(self, block_size: int) -> int:
 203          """Return the number of tokens consumed by one optimiser step.
 204  
 205          Args:
 206              block_size: The model's context length.
 207  
 208          Returns:
 209              ``batch_size * grad_accum_steps * block_size``.
 210          """
 211          return self.batch_size * self.grad_accum_steps * block_size
 212  
 213      def to_dict(self) -> dict[str, Any]:
 214          """Return this config as a plain dictionary."""
 215          return asdict(self)
 216  
 217      @classmethod
 218      def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
 219          """Build a config from a dictionary, ignoring unknown keys.
 220  
 221          Args:
 222              data: Mapping of field names to values.
 223  
 224          Returns:
 225              The constructed :class:`TrainConfig`.
 226          """
 227          known = {f.name for f in fields(cls)}
 228          return cls(**{k: v for k, v in data.items() if k in known})
 229  
 230  
 231  @dataclass
 232  class RunConfig:
 233      """A complete run: one architecture plus one optimisation setup."""
 234  
 235      model: GPTConfig
 236      train: TrainConfig
 237      notes: str = ""
 238  
 239      @classmethod
 240      def load(cls, path: str | Path) -> "RunConfig":
 241          """Read a run configuration from a YAML file.
 242  
 243          Args:
 244              path: Path to a YAML file with ``model:`` and ``train:`` sections.
 245  
 246          Returns:
 247              The parsed :class:`RunConfig`.
 248          """
 249          data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
 250          return cls(
 251              model=GPTConfig.from_dict(data["model"]),
 252              train=TrainConfig.from_dict(data["train"]),
 253              notes=data.get("notes", ""),
 254          )
 255  
 256      def save(self, path: str | Path) -> None:
 257          """Write this run configuration to a YAML file.
 258  
 259          Args:
 260              path: Destination path; parent directories are created.
 261          """
 262          path = Path(path)
 263          path.parent.mkdir(parents=True, exist_ok=True)
 264          payload = {
 265              "model": self.model.to_dict(),
 266              "train": self.train.to_dict(),
 267              "notes": self.notes,
 268          }
 269          path.write_text(
 270              yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
 271              encoding="utf-8",
 272          )
 273  
 274      def to_json(self) -> str:
 275          """Return this run configuration as an indented JSON string."""
 276          return json.dumps(
 277              {
 278                  "model": self.model.to_dict(),
 279                  "train": self.train.to_dict(),
 280                  "notes": self.notes,
 281              },
 282              indent=2,
 283          )
```


## hindi/train/train.py

```python
   1  """Pretraining loop with resumable checkpointing.
   2  
   3  Checkpoint resume is mandatory for this project: Colab sessions terminate
   4  without warning, and a run that cannot pick up where it stopped loses marks. A
   5  checkpoint written here contains model weights, optimizer state, learning-rate
   6  scheduler state, the current step, the token count consumed so far, the RNG
   7  state, and the full run configuration — everything needed to continue a run
   8  bit-for-bit rather than approximately.
   9  
  10  Model H and Model L are trained by separate invocations of this module against
  11  separate configs, corpora and output directories. Nothing is shared at runtime,
  12  and neither model is ever initialised from the other.
  13  """
  14  
  15  from __future__ import annotations
  16  
  17  import argparse
  18  import json
  19  import math
  20  import time
  21  from contextlib import nullcontext
  22  from pathlib import Path
  23  
  24  import numpy as np
  25  import torch
  26  
  27  from ..model.config import RunConfig
  28  from ..data.dataset import BatchSampler
  29  from ..model.transformer import GPT
  30  
  31  REPO_ROOT = Path(__file__).resolve().parents[2]
  32  
  33  
  34  class CosineWarmupSchedule:
  35      """Linear warmup followed by cosine decay to a floor learning rate.
  36  
  37      Warmup matters because AdamW's second-moment estimate is unreliable for the
  38      first few hundred steps; stepping at full learning rate before it settles is
  39      a common cause of early divergence. Cosine decay then spends most of the
  40      budget at a high rate and anneals smoothly at the end, which tends to beat a
  41      step schedule at fixed budget.
  42  
  43      Implemented explicitly rather than via ``torch.optim.lr_scheduler`` so that
  44      its state is a single integer and checkpoint resume is unambiguous.
  45      """
  46  
  47      def __init__(
  48          self,
  49          optimizer: torch.optim.Optimizer,
  50          peak_lr: float,
  51          min_lr: float,
  52          warmup_steps: int,
  53          max_steps: int,
  54      ) -> None:
  55          """Configure the schedule.
  56  
  57          Args:
  58              optimizer: Optimizer whose param groups will be updated.
  59              peak_lr: Learning rate reached at the end of warmup.
  60              min_lr: Floor reached at ``max_steps``.
  61              warmup_steps: Number of linear warmup steps.
  62              max_steps: Total steps in the run.
  63          """
  64          self.optimizer = optimizer
  65          self.peak_lr = peak_lr
  66          self.min_lr = min_lr
  67          self.warmup_steps = warmup_steps
  68          self.max_steps = max_steps
  69          self.last_step = 0
  70  
  71      def lr_at(self, step: int) -> float:
  72          """Return the learning rate for a given step.
  73  
  74          Args:
  75              step: Zero-based optimiser step.
  76  
  77          Returns:
  78              The learning rate to use at that step.
  79          """
  80          if step < self.warmup_steps:
  81              return self.peak_lr * (step + 1) / max(self.warmup_steps, 1)
  82          progress = (step - self.warmup_steps) / max(
  83              self.max_steps - self.warmup_steps, 1
  84          )
  85          progress = min(progress, 1.0)
  86          coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
  87          return self.min_lr + coeff * (self.peak_lr - self.min_lr)
  88  
  89      def step(self, step: int) -> float:
  90          """Set the learning rate on every param group for ``step``.
  91  
  92          Args:
  93              step: Zero-based optimiser step.
  94  
  95          Returns:
  96              The learning rate that was applied.
  97          """
  98          lr = self.lr_at(step)
  99          for group in self.optimizer.param_groups:
 100              group["lr"] = lr
 101          self.last_step = step
 102          return lr
 103  
 104      def state_dict(self) -> dict:
 105          """Return the schedule's resumable state."""
 106          return {
 107              "last_step": self.last_step,
 108              "peak_lr": self.peak_lr,
 109              "min_lr": self.min_lr,
 110              "warmup_steps": self.warmup_steps,
 111              "max_steps": self.max_steps,
 112          }
 113  
 114      def load_state_dict(self, state: dict) -> None:
 115          """Restore schedule *progress* from a checkpoint, keeping current shape.
 116  
 117          Only ``last_step`` is restored. The schedule's shape (peak rate, floor,
 118          warmup length, horizon) is deliberately taken from the live config
 119          instead of the checkpoint, because a resume that extends the run — the
 120          normal case when a Colab session dies and training is restarted with a
 121          larger ``max_steps`` — must follow the *new* horizon. Restoring the old
 122          one would leave ``step > max_steps``, pinning the learning rate at
 123          ``min_lr`` for the entire extension and making training look stalled.
 124  
 125          Any difference is reported rather than applied silently, since a
 126          changed schedule mid-run is something the training log should show.
 127  
 128          Args:
 129              state: A dict previously returned by :meth:`state_dict`.
 130          """
 131          self.last_step = state["last_step"]
 132          changed = {
 133              key: (state[key], getattr(self, key))
 134              for key in ("peak_lr", "min_lr", "warmup_steps", "max_steps")
 135              if key in state and state[key] != getattr(self, key)
 136          }
 137          for key, (old, new) in changed.items():
 138              print(f"  schedule {key}: checkpoint had {old}, using config's {new}")
 139  
 140  
 141  def resolve_device(requested: str) -> str:
 142      """Resolve ``"auto"`` to the best available device.
 143  
 144      Args:
 145          requested: ``"auto"``, ``"cuda"`` or ``"cpu"``.
 146  
 147      Returns:
 148          A concrete device string.
 149      """
 150      if requested != "auto":
 151          return requested
 152      return "cuda" if torch.cuda.is_available() else "cpu"
 153  
 154  
 155  def resolve_dtype(requested: str, device: str) -> torch.dtype:
 156      """Pick an autocast dtype the current device actually supports.
 157  
 158      bfloat16 is preferred over float16 where available: it has float32's
 159      exponent range, so it does not need loss scaling and cannot silently
 160      overflow to inf during training.
 161  
 162      Args:
 163          requested: ``"bfloat16"``, ``"float16"`` or ``"float32"``.
 164          device: Resolved device string.
 165  
 166      Returns:
 167          The dtype to autocast with.
 168      """
 169      if device == "cpu":
 170          return torch.float32
 171      if requested == "bfloat16" and not torch.cuda.is_bf16_supported():
 172          print("  bfloat16 unsupported on this GPU; falling back to float16")
 173          return torch.float16
 174      return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
 175          requested, torch.float32
 176      )
 177  
 178  
 179  class Trainer:
 180      """Runs one language's pretraining, with checkpointing and resume."""
 181  
 182      def __init__(self, run: RunConfig, resume: bool = True) -> None:
 183          """Set up the model, optimizer, schedule, data and output directory.
 184  
 185          Args:
 186              run: The full run configuration.
 187              resume: Whether to continue from the latest checkpoint if one exists.
 188          """
 189          self.run = run
 190          cfg, tcfg = run.model, run.train
 191  
 192          self.device = resolve_device(tcfg.device)
 193          self.dtype = resolve_dtype(tcfg.dtype, self.device)
 194          torch.manual_seed(tcfg.seed)
 195          np.random.seed(tcfg.seed)
 196  
 197          self.out_dir = REPO_ROOT / tcfg.out_dir
 198          self.out_dir.mkdir(parents=True, exist_ok=True)
 199          self.log_path = self.out_dir / "training_log.jsonl"
 200  
 201          self.model = GPT(cfg).to(self.device)
 202          self.optimizer = self.model.configure_optimizers(
 203              weight_decay=tcfg.weight_decay,
 204              learning_rate=tcfg.learning_rate,
 205              betas=(tcfg.beta1, tcfg.beta2),
 206              device_type="cuda" if self.device.startswith("cuda") else "cpu",
 207          )
 208          self.schedule = CosineWarmupSchedule(
 209              self.optimizer,
 210              peak_lr=tcfg.learning_rate,
 211              min_lr=tcfg.min_lr,
 212              warmup_steps=tcfg.warmup_steps,
 213              max_steps=tcfg.max_steps,
 214          )
 215          # GradScaler is only needed for float16; bfloat16 has enough exponent
 216          # range that gradients do not underflow.
 217          self.scaler = torch.amp.GradScaler(
 218              "cuda", enabled=(self.dtype == torch.float16)
 219          )
 220          self.autocast = (
 221              torch.amp.autocast(device_type="cuda", dtype=self.dtype)
 222              if self.device.startswith("cuda")
 223              else nullcontext()
 224          )
 225  
 226          self.data = BatchSampler(
 227              REPO_ROOT / tcfg.data_dir,
 228              block_size=cfg.block_size,
 229              device=self.device,
 230              seed=tcfg.seed,
 231          )
 232          self.eval_batches = self.data.fixed_batches(
 233              "val", tcfg.batch_size, tcfg.eval_iters
 234          ) if "val" in self.data.streams else []
 235  
 236          self.step = 0
 237          self.tokens_seen = 0
 238          self.best_val_loss = float("inf")
 239          if resume:
 240              self._maybe_resume()
 241  
 242          run.save(self.out_dir / "config.yaml")
 243  
 244      # -- checkpointing ----------------------------------------------------
 245  
 246      def _checkpoint_payload(self) -> dict:
 247          """Assemble everything needed to resume this run exactly.
 248  
 249          Returns:
 250              The dict that gets serialised to a checkpoint file.
 251          """
 252          return {
 253              "model": self.model.state_dict(),
 254              "optimizer": self.optimizer.state_dict(),
 255              "scheduler": self.schedule.state_dict(),
 256              "scaler": self.scaler.state_dict(),
 257              "step": self.step,
 258              "tokens_seen": self.tokens_seen,
 259              "best_val_loss": self.best_val_loss,
 260              "config": {
 261                  "model": self.run.model.to_dict(),
 262                  "train": self.run.train.to_dict(),
 263                  "notes": self.run.notes,
 264              },
 265              "parameter_count": self.model.parameter_count(),
 266              "torch_rng_state": torch.get_rng_state(),
 267              "numpy_rng_state": self.data.rng.bit_generator.state,
 268          }
 269  
 270      def save_checkpoint(self, tag: str | None = None) -> Path:
 271          """Write a checkpoint and prune old rolling checkpoints.
 272  
 273          Args:
 274              tag: Optional filename tag; defaults to the step number.
 275  
 276          Returns:
 277              Path to the checkpoint that was written.
 278          """
 279          name = f"ckpt_{tag}.pt" if tag else f"ckpt_step{self.step:06d}.pt"
 280          path = self.out_dir / name
 281          torch.save(self._checkpoint_payload(), path)
 282          # "latest.pt" is what resume looks for, so an interrupted run always has
 283          # an unambiguous entry point.
 284          torch.save(self._checkpoint_payload(), self.out_dir / "latest.pt")
 285          if tag is None:
 286              rolling = sorted(self.out_dir.glob("ckpt_step*.pt"))
 287              for stale in rolling[: -self.run.train.keep_last_n]:
 288                  stale.unlink(missing_ok=True)
 289          return path
 290  
 291      def _maybe_resume(self) -> None:
 292          """Restore state from ``latest.pt`` in the output directory, if present."""
 293          path = self.out_dir / "latest.pt"
 294          if not path.exists():
 295              print(f"  no checkpoint at {path}; starting from scratch")
 296              return
 297          ckpt = torch.load(path, map_location=self.device, weights_only=False)
 298          self.model.load_state_dict(ckpt["model"])
 299          self.optimizer.load_state_dict(ckpt["optimizer"])
 300          self.schedule.load_state_dict(ckpt["scheduler"])
 301          self.scaler.load_state_dict(ckpt["scaler"])
 302          self.step = ckpt["step"]
 303          self.tokens_seen = ckpt["tokens_seen"]
 304          self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
 305          if "torch_rng_state" in ckpt:
 306              torch.set_rng_state(ckpt["torch_rng_state"].cpu())
 307          if "numpy_rng_state" in ckpt:
 308              self.data.rng.bit_generator.state = ckpt["numpy_rng_state"]
 309          print(
 310              f"  resumed from {path} at step {self.step:,} "
 311              f"({self.tokens_seen/1e6:.1f}M tokens seen)"
 312          )
 313  
 314      # -- evaluation -------------------------------------------------------
 315  
 316      @torch.no_grad()
 317      def evaluate(self) -> float:
 318          """Average the loss over the fixed validation batches.
 319  
 320          Returns:
 321              Mean validation cross-entropy in nats, or ``nan`` if no validation
 322              split is available.
 323          """
 324          if not self.eval_batches:
 325              return float("nan")
 326          self.model.eval()
 327          losses = []
 328          for x, y in self.eval_batches:
 329              with self.autocast:
 330                  _, loss, _ = self.model(x, targets=y)
 331              losses.append(loss.item())
 332          self.model.train()
 333          return float(np.mean(losses))
 334  
 335      # -- logging ----------------------------------------------------------
 336  
 337      def _log(self, record: dict) -> None:
 338          """Append one record to the run's JSONL training log.
 339  
 340          The log is what the loss curves in the report are plotted from, so it is
 341          appended to rather than overwritten across resumes.
 342  
 343          Args:
 344              record: The record to append.
 345          """
 346          with open(self.log_path, "a", encoding="utf-8") as fh:
 347              fh.write(json.dumps(record) + "\n")
 348  
 349      # -- the loop ---------------------------------------------------------
 350  
 351      def train(self, max_minutes: float | None = None) -> None:
 352          """Run the training loop until the step budget or a time limit is hit.
 353  
 354          Args:
 355              max_minutes: Optional wall-clock budget. When set, the loop exits
 356                  cleanly (writing a checkpoint) before the limit, which is how a
 357                  Colab session is kept from being killed mid-write.
 358          """
 359          tcfg = self.run.train
 360          counts = self.model.parameter_count()
 361          tokens_per_step = tcfg.tokens_per_step(self.run.model.block_size)
 362  
 363          print(f"\n=== pretraining Model for {tcfg.language} ===")
 364          print(f"  device={self.device} dtype={self.dtype}")
 365          print(f"  parameters: {counts['total']:,} total, "
 366                f"{counts['non_embedding']:,} non-embedding")
 367          print(f"  data: {json.dumps(self.data.describe())}")
 368          print(f"  {tokens_per_step:,} tokens/step x {tcfg.max_steps:,} steps = "
 369                f"{tokens_per_step * tcfg.max_steps / 1e6:.1f}M tokens")
 370          print(f"  checkpoints -> {self.out_dir}\n")
 371  
 372          self.model.train()
 373          started = time.time()
 374          window_started = started
 375          window_loss = 0.0
 376          window_steps = 0
 377  
 378          while self.step < tcfg.max_steps:
 379              lr = self.schedule.step(self.step)
 380  
 381              self.optimizer.zero_grad(set_to_none=True)
 382              accum_loss = 0.0
 383              for _ in range(tcfg.grad_accum_steps):
 384                  x, y = self.data.sample("train", tcfg.batch_size)
 385                  with self.autocast:
 386                      _, loss, _ = self.model(x, targets=y)
 387                      # Scale so that accumulated gradients average, rather than
 388                      # sum, across the micro-batches.
 389                      loss = loss / tcfg.grad_accum_steps
 390                  self.scaler.scale(loss).backward()
 391                  accum_loss += loss.item()
 392  
 393              if tcfg.grad_clip > 0:
 394                  # Unscale before clipping, or the clip threshold would be
 395                  # applied to loss-scaled gradients.
 396                  self.scaler.unscale_(self.optimizer)
 397                  torch.nn.utils.clip_grad_norm_(
 398                      self.model.parameters(), tcfg.grad_clip
 399                  )
 400              self.scaler.step(self.optimizer)
 401              self.scaler.update()
 402  
 403              self.step += 1
 404              self.tokens_seen += tokens_per_step
 405              window_loss += accum_loss
 406              window_steps += 1
 407  
 408              if self.step % tcfg.log_interval == 0:
 409                  now = time.time()
 410                  mean_loss = window_loss / max(window_steps, 1)
 411                  elapsed = now - window_started
 412                  tps = tokens_per_step * window_steps / max(elapsed, 1e-9)
 413                  print(
 414                      f"  step {self.step:>6,}/{tcfg.max_steps:,}  "
 415                      f"loss {mean_loss:.4f}  ppl {math.exp(min(mean_loss, 20)):>8.1f}  "
 416                      f"lr {lr:.2e}  {tps/1e3:.1f}k tok/s  "
 417                      f"{self.tokens_seen/1e6:.0f}M seen",
 418                      flush=True,
 419                  )
 420                  self._log({
 421                      "step": self.step,
 422                      "train_loss": mean_loss,
 423                      "lr": lr,
 424                      "tokens_seen": self.tokens_seen,
 425                      "tokens_per_sec": tps,
 426                      "wall_seconds": now - started,
 427                  })
 428                  window_loss, window_steps = 0.0, 0
 429                  window_started = now
 430  
 431              if self.step % tcfg.eval_interval == 0:
 432                  val_loss = self.evaluate()
 433                  print(
 434                      f"  step {self.step:>6,}  VAL loss {val_loss:.4f}  "
 435                      f"ppl {math.exp(min(val_loss, 20)):.1f}",
 436                      flush=True,
 437                  )
 438                  self._log({
 439                      "step": self.step,
 440                      "val_loss": val_loss,
 441                      "val_ppl": math.exp(min(val_loss, 20)),
 442                      "tokens_seen": self.tokens_seen,
 443                  })
 444                  if val_loss < self.best_val_loss:
 445                      self.best_val_loss = val_loss
 446                      self.save_checkpoint(tag="best")
 447  
 448              if self.step % tcfg.checkpoint_interval == 0:
 449                  self.save_checkpoint()
 450  
 451              if max_minutes is not None and (time.time() - started) / 60 >= max_minutes:
 452                  print(f"\n  wall-clock budget of {max_minutes} min reached")
 453                  break
 454  
 455          self.save_checkpoint(tag="final")
 456          print(
 457              f"\n=== done: step {self.step:,}, "
 458              f"{self.tokens_seen/1e6:.1f}M tokens, "
 459              f"best val loss {self.best_val_loss:.4f} "
 460              f"({time.time() - started:.0f}s this session) ==="
 461          )
 462  
 463  
 464  def main() -> None:
 465      """Parse arguments and run pretraining for one language."""
 466      parser = argparse.ArgumentParser(description="Pretrain one monolingual LM.")
 467      parser.add_argument("--config", required=True, help="Path to a run YAML.")
 468      parser.add_argument(
 469          "--no-resume",
 470          action="store_true",
 471          help="Ignore any existing checkpoint and start over.",
 472      )
 473      parser.add_argument(
 474          "--max-minutes",
 475          type=float,
 476          default=None,
 477          help="Stop cleanly after this many minutes (for Colab session limits).",
 478      )
 479      parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps.")
 480      parser.add_argument("--device", default=None, help="Override the device.")
 481      parser.add_argument(
 482          "--data-dir",
 483          default=None,
 484          help="Override data_dir. Needed on Kaggle, where the packed streams are "
 485          "mounted read-only under /kaggle/input/<dataset>/ rather than in the repo.",
 486      )
 487      parser.add_argument(
 488          "--out-dir",
 489          default=None,
 490          help="Override out_dir. On Kaggle this should point inside "
 491          "/kaggle/working/, the only writable location that survives the session.",
 492      )
 493      parser.add_argument(
 494          "--dtype",
 495          default=None,
 496          choices=["bfloat16", "float16", "float32"],
 497          help="Override the autocast dtype. Kaggle's P100 and T4 GPUs predate "
 498          "Ampere and have no bfloat16 support; the trainer falls back to float16 "
 499          "automatically, but this makes the choice explicit in the run config.",
 500      )
 501      args = parser.parse_args()
 502  
 503      run = RunConfig.load(args.config)
 504      if args.max_steps is not None:
 505          run.train.max_steps = args.max_steps
 506      if args.device is not None:
 507          run.train.device = args.device
 508      if args.data_dir is not None:
 509          run.train.data_dir = args.data_dir
 510      if args.out_dir is not None:
 511          run.train.out_dir = args.out_dir
 512      if args.dtype is not None:
 513          run.train.dtype = args.dtype
 514  
 515      Trainer(run, resume=not args.no_resume).train(max_minutes=args.max_minutes)
 516  
 517  
 518  if __name__ == "__main__":
 519      main()
```


## hindi/configs/model.yaml

```yaml
   1  # Model H — Hindi. Phase 2 pretraining configuration.
   2  #
   3  # Independent of Model L in every graded respect: its own corpus
   4  # (hindi/data/packed/), its own SentencePiece Unigram tokenizer and 8,000-token
   5  # vocabulary from Phase 1, and its own weights. Only source code is shared.
   6  
   7  model:
   8    # Phase 1 tokenizer: hindi/tokenizer/hindi_unigram_8000.model
   9    vocab_size: 8000
  10  
  11    # Context length. Learned absolute positional embeddings make this a hard
  12    # ceiling — there is no embedding row for position 512 or beyond, so the model
  13    # cannot be evaluated on longer sequences without retraining that table.
  14    # 512 keeps the T x T attention matrix at 262k entries per head, which is what
  15    # makes a ~500M-token budget affordable inside a Colab session.
  16    block_size: 512
  17  
  18    # Depth/width trade-off. At n_embd=512 the block stack costs ~3.15M parameters
  19    # per layer, so 6 layers plus the 4.1M-parameter embedding table lands at
  20    # 23,273,472 total — just under the ~25M target. Seven layers would overshoot
  21    # to 26.4M. Depth was spent on 6 blocks rather than a wider 4-block model
  22    # because attention needs several rounds of composition to build multi-hop
  23    # dependencies, and because a wider model would grow the feed-forward cost
  24    # quadratically for the same parameter budget.
  25    #
  26    # Note that n_layer/n_head/n_embd/d_ff are IDENTICAL to Model L's. The
  27    # non-embedding parameter count is therefore identical (18,915,328) for both
  28    # models, so any difference in results is attributable to data and tokenizer
  29    # rather than to model capacity.
  30    n_layer: 6
  31    n_head: 8
  32    n_embd: 512
  33    d_ff: 2048
  34  
  35    dropout: 0.1
  36    bias: true
  37  
  38    # Ties the output projection to the input embedding, saving 8000 x 512 =
  39    # 4,096,000 parameters (~15% of the model) that would otherwise be spent on a
  40    # second representation of the same vocabulary.
  41    tie_weights: true
  42  
  43    # Learned absolute. Chosen over sinusoidal because the model is trained and
  44    # evaluated at a single fixed context length, where the extrapolation benefit
  45    # of sinusoidal encodings does not apply, and learned positions let the model
  46    # allocate resolution where the data needs it. Set to "none" to reproduce the
  47    # optional bonus ablation.
  48    pos_encoding: learned
  49  
  50    init_std: 0.02
  51  
  52  train:
  53    language: hindi
  54    data_dir: hindi/data/packed
  55    out_dir: checkpoints/hindi
  56  
  57    # 15,000 steps x 32,768 tokens/step = 491.5M tokens, i.e. approximately one
  58    # epoch over the 485M-token Hindi train split and on target for the spec's
  59    # ~500M-token budget. Model L uses the same step count and the same
  60    # tokens-per-step, so the two models are trained on equal token budgets.
  61    max_steps: 15000
  62  
  63    # Effective batch = 32 x 2 x 512 = 32,768 tokens. batch_size is the micro-batch
  64    # that must fit in VRAM; grad_accum_steps decouples the effective batch from
  65    # the available memory. Lower batch_size and raise grad_accum_steps to keep
  66    # the same effective batch on a smaller GPU.
  67    batch_size: 32
  68    grad_accum_steps: 2
  69  
  70    learning_rate: 6.0e-4
  71    min_lr: 6.0e-5
  72    warmup_steps: 500
  73    weight_decay: 0.1
  74    beta1: 0.9
  75    beta2: 0.95
  76    grad_clip: 1.0
  77  
  78    eval_interval: 500
  79    eval_iters: 100
  80    log_interval: 20
  81  
  82    # Checkpoints carry weights, optimizer state, scheduler state, step and this
  83    # config, so an interrupted Colab session resumes exactly where it stopped.
  84    checkpoint_interval: 500
  85    keep_last_n: 2
  86  
  87    seed: 20260902
  88    device: auto
  89    dtype: bfloat16
  90    compile_model: false
  91  
  92  notes: >
  93    Model H (higher-resource). Corpus: 505,852,028 tokens across 704,101 documents,
  94    30.66% from manual collection. Tokenizer: SentencePiece Unigram, vocab 8,000,
  95    test fertility measured in Phase 1.
```


## nepali/configs/model.yaml

```yaml
   1  # Model L — Nepali. Phase 2 pretraining configuration.
   2  #
   3  # Independent of Model H in every graded respect: its own corpus
   4  # (nepali/data/packed/), its own SentencePiece BPE tokenizer and 10,000-token
   5  # vocabulary from Phase 1, and its own weights. Only source code is shared.
   6  
   7  model:
   8    # Phase 1 tokenizer: nepali/tokenizer/nepali_bpe.model
   9    # Larger than Hindi's 8,000 because Nepali's Phase 1 vocabulary sweep found no
  10    # fertility elbow below 10,000 — fertility was still improving by more than 3%
  11    # at every step up to the top of the swept range.
  12    vocab_size: 10000
  13  
  14    # Same context length as Model H, so perplexity and bits-per-byte are computed
  15    # over comparably sized windows.
  16    block_size: 512
  17  
  18    # Identical depth and width to Model H by design. The non-embedding parameter
  19    # count is therefore exactly the same (18,915,328); the models differ only in
  20    # the vocabulary-sized embedding layer (10,000 x 512 here vs. 8,000 x 512),
  21    # giving 24,297,472 total against Hindi's 23,273,472. Holding capacity fixed
  22    # is what lets the Phase 3 report attribute any H-vs-L gap to data scale,
  23    # data quality and tokenizer fertility rather than to model size.
  24    n_layer: 6
  25    n_head: 8
  26    n_embd: 512
  27    d_ff: 2048
  28  
  29    dropout: 0.1
  30    bias: true
  31  
  32    # Saves 10000 x 512 = 5,120,000 parameters (~21% of the model here — a larger
  33    # share than for Hindi, precisely because the vocabulary is larger).
  34    tie_weights: true
  35  
  36    pos_encoding: learned
  37    init_std: 0.02
  38  
  39  train:
  40    language: nepali
  41    data_dir: nepali/data/packed
  42    out_dir: checkpoints/nepali
  43  
  44    # Same budget as Model H: 15,000 x 32,768 = 491.5M tokens, against a 520M-token
  45    # train split. Equal token budgets across the two models is deliberate.
  46    max_steps: 15000
  47  
  48    batch_size: 32
  49    grad_accum_steps: 2
  50  
  51    learning_rate: 6.0e-4
  52    min_lr: 6.0e-5
  53    warmup_steps: 500
  54    weight_decay: 0.1
  55    beta1: 0.9
  56    beta2: 0.95
  57    grad_clip: 1.0
  58  
  59    eval_interval: 500
  60    eval_iters: 100
  61    log_interval: 20
  62  
  63    checkpoint_interval: 500
  64    keep_last_n: 2
  65  
  66    # Different seed from Model H's, so the two runs are independent draws rather
  67    # than the same run with different data.
  68    seed: 20260903
  69    device: auto
  70    dtype: bfloat16
  71    compile_model: false
  72  
  73  notes: >
  74    Model L (lower-resource, from the spec's allowed list). Corpus: 635,204,333
  75    tokens across 1,918,265 documents, 26.41% from manual collection. Tokenizer:
  76    SentencePiece BPE, vocab 10,000, test fertility 1.4672, unknown-token rate 0.
```
