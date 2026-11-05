"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import bisect
from collections import deque
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch._higher_order_ops import foreach_map

from kernels import get_kernel
cap = torch.cuda.get_device_capability()
# varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface

from prepare import (
    MAX_SEQ_LEN, TIME_BUDGET, Tokenizer as BaseTokenizer, evaluate_bpb, evaluate_olmes,
    _document_batches,
)

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

class Tokenizer(BaseTokenizer):
    def encode(self, text, prepend=None, num_threads=8):
        # Avoid per-call thread-pool setup for small inference batches.
        # https://github.com/openai/tiktoken/blob/0.14.0/tiktoken/core.py
        if isinstance(text, list) and len(text) <= 4 and num_threads == 8:
            ids = [self.enc.encode_ordinary(item) for item in text]
            if prepend is not None:
                prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
                for row in ids:
                    row.insert(0, prepend_id)
            return ids
        return super().encode(text, prepend=prepend, num_threads=num_threads)

    __call__ = encode


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 8
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 512
    window_pattern: str = "SSSL"
    narrow_first_layer: bool = True
    float32_embedding_master: bool = True
    share_value_embeddings: bool = True
    value_silu: bool = True
    training_loss_chunk_size: int = 16384


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Use Value Embeddings in the deeper half of the transformer layers."""
    return layer_idx >= n_layer // 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # The narrow block uses a prefix of the trunk's rotary frequencies.
    cos, sin = cos[..., :d], sin[..., :d]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


@torch.compile(dynamic=True, fullgraph=True)
def prepare_qk(q, k, cos, sin):
    # Keep capture local to model calculations when prompt lengths vary.
    # https://docs.pytorch.org/docs/2.9/torch.compiler_dynamic_shapes.html
    q = norm(apply_rotary_emb(q, cos, sin)) * 1.2
    k = norm(apply_rotary_emb(k, cos, sin)) * 1.2
    return q, k


@torch.compile(dynamic=True, fullgraph=True)
def squared_relu(x):
    return F.relu(x).square()


@torch.compile(dynamic=True, fullgraph=True)
def softcap_logits(logits):
    return 15 * torch.tanh(logits.float() / 15)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx, n_embd=None):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd if n_embd is None else n_embd
        self.value_silu = config.value_silu
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.head_dim % 2 == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = min(32, self.n_embd)
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def project_qkv(self, x):
        # Keep separate Muon parameters while sharing the projection GEMM.
        weight = torch.cat((self.c_q.weight, self.c_k.weight, self.c_v.weight), dim=0)
        widths = (self.n_head * self.head_dim, self.n_kv_head * self.head_dim, self.n_kv_head * self.head_dim)
        return F.linear(x, weight).split(widths, dim=-1)

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q, k, v = self.project_qkv(x)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # GatedV: https://arxiv.org/abs/2611.02145v2
        # Self-gate the complete value stream, including value embeddings, before aggregation.
        if self.value_silu:
            v = F.silu(v)

        cos, sin = cos_sin
        q, k = prepare_qk(q, k, cos, sin)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config, n_embd=None):
        super().__init__()
        width = config.n_embd if n_embd is None else n_embd
        self.c_fc = nn.Linear(width, 4 * width, bias=False)
        self.c_proj = nn.Linear(4 * width, width, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = squared_relu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx, n_embd=None):
        super().__init__()
        width = config.n_embd if n_embd is None else n_embd
        assert width == config.n_embd or 2 * width == config.n_embd
        self.widen_output = width != config.n_embd
        self.attn = CausalSelfAttention(config, layer_idx, width)
        self.mlp = MLP(config, width)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        if self.widen_output:
            x = torch.cat((x, x), dim=-1)
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.training_loss_chunk_size >= 0
        self.window_sizes = self._compute_window_sizes(config)
        # Narrow input block: https://github.com/karpathy/nanochat/pull/807
        assert not config.narrow_first_layer or config.n_embd % 2 == 0
        embedding_width = config.n_embd // 2 if config.narrow_first_layer else config.n_embd
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, embedding_width),
            "h": nn.ModuleList([Block(config, i, embedding_width if i == 0 else config.n_embd)
                                for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Smear/backout: https://github.com/karpathy/nanochat/commit/a825e63f81e62e2e9fd38655e9b2e39417620545
        self.smear_channels = min(24, embedding_width)
        self.smear_gate = nn.Linear(self.smear_channels, 1, bias=False)
        self.smear_gain = nn.Parameter(torch.zeros(1))
        self.backout_gain = nn.Parameter(torch.full((1,), 0.2))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        # Mirror the selected layers' value tables while keeping their projected values and gates.
        # https://github.com/KellerJordan/modded-nanogpt/tree/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_1_short/2024-12-08_UNetValueEmbedsTweaks
        ve_layers = [i for i in range(config.n_layer) if has_ve(i, config.n_layer)]
        self.value_embed_keys = {}
        self.value_embeds = nn.ModuleDict()
        for position, layer in enumerate(ve_layers):
            attention = self.transformer.h[layer].attn
            width = attention.n_kv_head * attention.head_dim
            pair = min(position, len(ve_layers) - 1 - position)
            key = f"shared_{pair}_{width}" if config.share_value_embeddings else str(layer)
            self.value_embed_keys[layer] = key
            if key not in self.value_embeds:
                self.value_embeds[key] = nn.Embedding(config.vocab_size, width)
        self.value_embed_last_layer = {key: layer for layer, key in self.value_embed_keys.items()}
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Initialization: https://github.com/karpathy/nanochat/commit/a825e63f81e62e2e9fd38655e9b2e39417620545
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        for i in range(self.config.n_layer):
            fraction = i / max(self.config.n_layer - 1, 1)
            self.resid_lambdas[i] = 1.15 - 0.10 * fraction
            self.x0_lambdas[i] = 0.20 - 0.15 * fraction
        self.smear_gain.zero_()
        self.backout_gain.fill_(0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # FP32 master weights: https://arxiv.org/abs/1710.03740
        embedding_dtype = torch.float32 if self.config.float32_embedding_master else torch.bfloat16
        self.transformer.wte.to(dtype=embedding_dtype)
        for ve in self.value_embeds.values():
            ve.to(dtype=embedding_dtype)

    # RoPE base: https://github.com/karpathy/nanochat/commit/6ed7d1d82cee16c2e26f45d559ad3338447a6c1b
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        short_window = config.sequence_len // 4
        # Let L layers attend to the supplied prefix, including longer inputs.
        char_to_window = {"L": (-1, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (-1, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gain.numel() + self.backout_gain.numel())
        t = self.config.sequence_len
        attn_flops = 0
        for block, window_size in zip(self.transformer.h, self.window_sizes, strict=True):
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * block.attn.n_head * block.attn.head_dim * effective_seq
        head_recompute_flops = (2 * self.lm_head.weight.numel()
                                if self.config.training_loss_chunk_size > 0 else 0)
        return 6 * (nparams - nparams_exclude) + attn_flops + head_recompute_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = (self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                   self.smear_gate.weight.numel() + self.smear_gain.numel() + self.backout_gain.numel())
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(parameter.numel() for parameter in self.parameters())
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.008, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        mixing_params = [self.smear_gate.weight, self.smear_gain, self.backout_gain]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(mixing_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=mixing_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _loss_sum(self, hidden, targets):
        logits = softcap_logits(self.lm_head(hidden))
        return F.cross_entropy(logits, targets, ignore_index=-1, reduction='sum')

    def _training_loss(self, hidden, targets):
        # Bound output-layer activations by recomputing one token chunk during backward.
        # https://docs.pytorch.org/docs/2.9/checkpoint.html
        hidden = hidden.reshape(-1, hidden.size(-1))
        targets = targets.reshape(-1)
        chunk_size = self.config.training_loss_chunk_size
        loss = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, hidden.size(0), chunk_size):
            stop = start + chunk_size
            loss = loss + torch.utils.checkpoint.checkpoint(
                self._loss_sum, hidden[start:stop], targets[start:stop],
                use_reentrant=False, preserve_rng_state=False,
            )
        return loss / (targets != -1).sum()

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        activation_dtype = (torch.get_autocast_dtype(idx.device.type)
                            if torch.is_autocast_enabled(idx.device.type) else self.lm_head.weight.dtype)
        x = self.transformer.wte(idx).to(activation_dtype)
        x = norm(x)
        if T > 1:
            gate = torch.sigmoid(self.smear_gate(x[:, 1:, :self.smear_channels]))
            mixed = x[:, 1:] + self.smear_gain.to(x.dtype) * gate * x[:, :-1]
            x = torch.cat((x[:, :1], mixed), dim=1)
        x0 = x
        middle = None
        value_cache = {}
        for i, block in enumerate(self.transformer.h):
            if x0.size(-1) < x.size(-1):
                x0 = torch.cat((x0, x0), dim=-1)
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve_key = self.value_embed_keys.get(i)
            ve = None
            if ve_key is not None:
                cache_key = (ve_key, x.dtype)
                if cache_key not in value_cache:
                    value_cache[cache_key] = self.value_embeds[ve_key](idx).to(x.dtype)
                ve = value_cache[cache_key]
            x = block(x, ve, cos_sin, self.window_sizes[i])
            if ve_key is not None and self.value_embed_last_layer[ve_key] == i:
                value_cache.pop(cache_key)
            if i == self.config.n_layer // 2:
                middle = x
        value_cache.clear()
        if middle is not None:
            x = x - self.backout_gain.to(x.dtype) * middle
        x = norm(x)

        if (targets is not None and reduction == 'mean' and self.training
                and torch.is_grad_enabled() and self.config.training_loss_chunk_size > 0):
            return self._training_loss(x, targets)

        logits = softcap_logits(self.lm_head(x))

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits


class TailEMA:
    """Average late training iterates, then return one fixed blend for both evaluations."""

    def __init__(self, model, start, tau, blend):
        assert 0 <= start < 1 and math.isfinite(tau) and tau > 0 and 0 <= blend <= 1
        self.params = [p for p in model.parameters() if p is not model.transformer.wte.weight]
        self.start, self.tau, self.blend = start, tau, blend
        self.average = None
        self.last_progress = None
        self.num_updates = 0

    @torch.no_grad()
    def update(self, progress):
        if progress < self.start:
            return
        if self.average is None:
            self.average = [p.detach().float().clone() for p in self.params]
        else:
            delta = progress - self.last_progress
            if delta < 0:
                raise ValueError("Tail EMA requires nondecreasing training progress.")
            weight = -math.expm1(-delta / self.tau)
            torch._foreach_lerp_(self.average, [p.detach().float() for p in self.params], weight)
        self.last_progress = progress
        self.num_updates += 1

    @torch.no_grad()
    def apply(self):
        if self.average is None:
            raise RuntimeError("The declared tail average was not accumulated.")
        device = self.params[0].device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for parameter, average in zip(self.params, self.average, strict=True):
            parameter.copy_(parameter.detach().float().lerp(average, self.blend))
        self.average = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter() - started


class LengthIndexedDocuments:
    """Select the native best fit while preserving first-arrival ties."""

    def __init__(self):
        self.lengths = []
        self.buckets = {}
        self.count = 0

    def __len__(self):
        return self.count

    def extend(self, documents):
        for document in documents:
            length = len(document)
            if length not in self.buckets:
                bisect.insort(self.lengths, length)
                self.buckets[length] = deque()
            self.buckets[length].append(document)
            self.count += 1

    def pop_best(self, remaining):
        index = max(0, bisect.bisect_right(self.lengths, remaining) - 1)
        length = self.lengths[index]
        bucket = self.buckets[length]
        document = bucket.popleft()
        self.count -= 1
        if not bucket:
            self.lengths.pop(index)
            del self.buckets[length]
        return document


def make_training_dataloader(tokenizer, B, T, buffer_size=1000):
    # Keep prepare.py's document stream, refill boundary, cropping and shifted targets.
    # Per-length FIFO queues preserve its first-match tie rule without scanning every document.
    row_capacity = T + 1
    batches = _document_batches("train")
    bos_token = tokenizer.get_bos_token_id()
    documents = LengthIndexedDocuments()
    epoch = 1
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    row_values = row_buffer.numpy()
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)
    transfer_done = None
    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(documents) < buffer_size:
                    texts, epoch = next(batches)
                    documents.extend(tokenizer.encode(texts, prepend=bos_token))
                remaining = row_capacity - pos
                document = documents.pop_best(remaining)
                if len(document) > remaining:
                    document = document[:remaining]
                row_values[row_idx, pos:pos + len(document)] = document
                pos += len(document)
        # Pinned host storage must remain unchanged until its prior H2D copy completes.
        if transfer_done is not None:
            transfer_done.synchronize()
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        if gpu_buffer.is_cuda:
            if transfer_done is None:
                transfer_done = torch.cuda.Event()
            transfer_done.record()
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

def adamw_parameter_update(p, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    return p * (1 - lr_t * wd_t) - (exp_avg / denom) * step_size


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(params, grads, exp_avgs, exp_avg_sqs, steps, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    # Expose independent updates to horizontal fusion without stacking parameter storage.
    # https://docs.pytorch.org/tutorials/recipes/foreach_map.html
    means = foreach_map(torch.lerp, exp_avgs, grads, 1 - beta1_t)
    squares = foreach_map(lambda state, grad, beta2: torch.lerp(state, grad.square(), 1 - beta2),
                          exp_avg_sqs, grads, beta2_t)
    updated = foreach_map(adamw_parameter_update, params, means, squares, steps,
                          lr_t, beta1_t, beta2_t, eps_t, wd_t)
    torch._foreach_copy_(exp_avgs, means)
    torch._foreach_copy_(exp_avg_sqs, squares)
    torch._foreach_copy_(params, updated)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        params, grads, exp_avgs, exp_avg_sqs, steps = [], [], [], [], []
        for p in group['params']:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['step_t'] = torch.tensor(0.0, dtype=torch.float32, device="cpu")
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            state['step_t'].fill_(state['step'])
            params.append(p)
            grads.append(p.grad)
            exp_avgs.append(state['exp_avg'])
            exp_avg_sqs.append(state['exp_avg_sq'])
            steps.append(state['step_t'])
        if not params:
            return
        self._adamw_lr_t.fill_(group['lr'])
        self._adamw_beta1_t.fill_(group['betas'][0])
        self._adamw_beta2_t.fill_(group['betas'][1])
        self._adamw_eps_t.fill_(group['eps'])
        self._adamw_wd_t.fill_(group['weight_decay'])
        adamw_step_fused(params, grads, exp_avgs, exp_avg_sqs, steps,
                        self._adamw_lr_t, self._adamw_beta1_t, self._adamw_beta2_t,
                        self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)


def scale_adam_second_moments(optimizer, batch_ratio):
    """Keep Adam's nominal second-moment half-life in tokens after a batch change."""
    # https://arxiv.org/abs/2507.07101v2
    if not math.isfinite(batch_ratio) or batch_ratio <= 0:
        raise ValueError("The batch ratio must be finite and positive.")
    for group in optimizer.param_groups:
        if group['kind'] == 'adamw':
            beta1, beta2 = group['betas']
            group['betas'] = (beta1, beta2 ** batch_ratio)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=quarter context

# Optimization
TOTAL_BATCH_SIZE = 2**17 # one device batch, ~131K tokens per optimizer step
BATCH_REFERENCE_SIZE = 2**18 # parent batch for Adam variance, Muon decay and warmup
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.008  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.28     # reference-batch cautious weight decay for Muon
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.65   # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.05    # final LR as fraction of initial
# Tail-EMA adaptation: https://github.com/KellerJordan/modded-nanogpt/pull/398
TAIL_EMA_START = 0.80   # fraction of timed training before averaging begins
TAIL_EMA_TAU = 0.075    # time constant as a fraction of the fixed training budget
TAIL_EMA_BLEND = 0.60   # weight of EMA in the single returned model

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 64   # per-device batch size (reduce if OOM)

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY * TOTAL_BATCH_SIZE / BATCH_REFERENCE_SIZE,
)
scale_adam_second_moments(optimizer, TOTAL_BATCH_SIZE / BATCH_REFERENCE_SIZE)

tail_ema = TailEMA(model, TAIL_EMA_START, TAIL_EMA_TAU, TAIL_EMA_BLEND)
model = torch.compile(model, dynamic=False)

train_loader = make_training_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN)
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

# Momentum recipe: https://github.com/karpathy/nanochat/commit/a825e63f81e62e2e9fd38655e9b2e39417620545
def get_muon_momentum(step, progress):
    warmup_tokens = 400 * BATCH_REFERENCE_SIZE
    processed_tokens = step * TOTAL_BATCH_SIZE
    if processed_tokens < warmup_tokens:
        frac = processed_tokens / warmup_tokens
        return (1 - frac) * 0.85 + frac * 0.97
    warmdown_start = 1.0 - WARMDOWN_RATIO
    if WARMDOWN_RATIO > 0 and progress >= warmdown_start:
        frac = (progress - warmdown_start) / WARMDOWN_RATIO
        return (1 - frac) * 0.97 + frac * 0.90
    return 0.97

def get_weight_decay(progress):
    # Preserve nominal decay exposure per token at a fixed learning rate and progress.
    # Cautious masks and the actual training trajectory can still change.
    # https://arxiv.org/html/2610.08975v1
    batch_ratio = TOTAL_BATCH_SIZE / BATCH_REFERENCE_SIZE
    return WEIGHT_DECAY * batch_ratio * 0.5 * (1 + math.cos(math.pi * progress))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step, progress)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)
    tail_ema.update(progress)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE
total_training_time += tail_ema.apply()

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
olmes = evaluate_olmes(model, tokenizer)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = max(torch.cuda.max_memory_allocated() / 1024 / 1024,
                   olmes["combined_peak_vram_mb"])

print("---")
for name in ("qa_bpb", "math_bpb", "code_bpb"):
    print(f"{name}: {olmes['suites'][name]:.17g}")
print(f"val_bpb:          {val_bpb:.17g}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
