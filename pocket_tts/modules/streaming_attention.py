from typing import Dict, Optional

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.stateful_module import StatefulModule


class KVCacheResult:
    """Result from KV cache completion with keys, values, and position info."""

    __slots__ = ("keys", "values", "positions")

    def __init__(self, keys: torch.Tensor, values: torch.Tensor, positions: torch.Tensor):
        self.keys = keys
        self.values = values
        self.positions = positions

    @staticmethod
    def from_kv(keys: torch.Tensor, values: torch.Tensor) -> "KVCacheResult":
        """Create from K/V tensors without cached history."""
        B, H, T, D = keys.shape
        assert tuple(values.shape[:-1]) == (B, H, T)
        positions = torch.arange(T, device=keys.device, dtype=torch.long)
        return KVCacheResult(keys, values, positions.expand(B, -1))


def _complete_ring_buffer(
    cache: torch.Tensor, end_offset: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> KVCacheResult:
    """Complete KV cache using ring buffer for sliding window attention.

    Args:
        cache: Shape [2, B, H, capacity, D] - ring buffer for keys and values
        end_offset: Shape [B] - current write position for each batch
        k: Shape [B, H, T, D] - new keys to add
        v: Shape [B, H, T, D] - new values to add

    Returns:
        KVCacheResult with full keys, values, and position info
    """
    capacity = cache.shape[3]
    assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
    B, H, T, D = k.shape
    assert T > 0

    # Calculate indices for ring buffer insertion
    indexes = torch.arange(T, device=end_offset.device, dtype=end_offset.dtype)
    indexes = indexes + end_offset.view(-1, 1)
    indexes = indexes % capacity

    # Scatter new K/V into cache
    this_indexes = indexes.view(B, 1, T, 1).expand(-1, H, T, D)
    cache[0].scatter_(2, this_indexes, k)
    cache[1].scatter_(2, this_indexes, v)

    keys = cache[0]
    values = cache[1]

    # Calculate positions for attention masking
    indexes = torch.arange(capacity, device=end_offset.device, dtype=torch.long)
    last_offset = end_offset.view(-1, 1) + T - 1
    end_index = last_offset % capacity
    delta = indexes - end_index
    positions = torch.where(delta <= 0, last_offset + delta, last_offset + delta - capacity)

    end_offset[:] = end_offset + T
    invalid = indexes >= end_offset.view(-1, 1)
    positions = torch.where(invalid, torch.full_like(positions, -1), positions)

    return KVCacheResult(keys, values, positions)


def _complete_append_buffer(
    cache: torch.Tensor, current_end: int, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complete KV cache using simple append for full-sequence attention.

    Args:
        cache: Shape [2, B, capacity, H, D] - buffer for keys and values
        current_end: Current length of cached sequence
        k: Shape [B, T, H, D] - new keys to add
        v: Shape [B, T, H, D] - new values to add

    Returns:
        Tuple of (keys, values) including all cached history
    """
    cache[0, :, current_end : current_end + k.shape[1]] = k
    cache[1, :, current_end : current_end + v.shape[1]] = v
    valid = cache[:, :, : current_end + k.shape[1]]
    return valid[0], valid[1]


def _materialize_causal_mask(
    shape: tuple[int, ...], shift: int, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Create a causal attention mask."""
    dtype = torch.float32
    num_queries, num_keys = shape[-2:]
    shift = num_keys - num_queries
    tensor = torch.full(shape, dtype=dtype, fill_value=1, device=device)
    mask = torch.tril(tensor, diagonal=shift).to(dtype)
    mask = torch.log(mask)
    return mask.to(dtype)


class StreamingMultiheadAttention(StatefulModule):
    """Unified streaming multi-head attention with optional context window support.

    This class unifies the previous MimiStreamingMultiheadAttention and
    StreamingMultiheadAttention implementations. When context is specified,
    it uses a ring buffer for sliding window attention. Otherwise, it uses
    full-sequence attention with simple append caching.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        rope: Rotary position embedding module.
        context: Optional context window size for sliding window attention.
            If None, uses full-sequence attention.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, rope: RotaryEmbedding, context: int | None = None
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rope = rope
        self.context = context

        out_dim = 3 * embed_dim
        self.in_proj = nn.Linear(embed_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    # Remove type hints to prevent beartype wrapping (fixes JIT undefined torch)
    def init_state(self, batch_size: int, sequence_length: int) -> Dict[str, torch.Tensor]:
        """Initialize the state in a JIT-compatible way."""
        dim_per_head = self.embed_dim // self.num_heads
        ref = self.in_proj.weight

        if self.context is not None:
            # Ring buffer mode for sliding window attention
            return {
                "offset": ref.new_zeros((batch_size,)).long(),
                "cache": ref.new_zeros(
                    (2, batch_size, self.num_heads, sequence_length, dim_per_head)
                ),
                "end_offset": ref.new_zeros((batch_size,)).long(),
            }
        else:
            # Use reference tensor to create new tensors without accessing 'torch' global
            # which is hidden by beartype wrapper in JIT

            # current_end needs to be long. new_zeros creates float (matches ref).
            # We assume .long() is JIT supported.
            return {
                "current_end": ref.new_zeros(int(batch_size)).long(),
                "cache": ref.new_full(
                    [2, int(batch_size), int(sequence_length), self.num_heads, dim_per_head],
                    float("nan"),
                ),
            }

    def increment_step(self, state: dict, increment: int = 1):
        if self.context is not None:
            state["offset"] += increment
        else:
            new_size = state["current_end"].shape[0] + increment
            state["current_end"] = torch.zeros((new_size,)).to(state["current_end"].device)

    def forward(
        self, query: torch.Tensor, model_state: Optional[Dict[str, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        if model_state is None:
            # Create a temporary state for this forward pass
            B, T = query.shape[:2]
            # Use T as capacity if context is None (append mode), otherwise use context relative to T?
            # Actually init_state expects 'sequence_length' which dictates buffer size.
            # If context is None, init_state builds a buffer of size 'sequence_length'.
            # If context is set, init_state builds a ring buffer of that size (ignoring sequence_length arg usually? No check init_state)

            # Explicitly capture self.context for JIT type refinement
            context_val = self.context
            if context_val is not None:
                capacity = max(T, context_val)
            else:
                capacity = T

            if context_val is not None:
                if T > context_val:
                    # We are processing a chunk larger than window.
                    # Buffer will hold T. attn_bias handles masking.
                    pass

            # Initialize state
            state_dict = self.init_state(B, capacity)

            # No need to move, init_state uses ref.new_...

            # The structure of our state dict is simple for this module
            state = state_dict
        else:
            state = self.get_state(model_state)

        B, T = query.shape[:2]

        projected = self.in_proj(query)

        # Calculate shapes
        d = self.embed_dim // self.num_heads

        # Check projected dims
        if len(projected.shape) == 2:
            # Handle case where B might be folded or query was Rank 2
            pass

        packed = projected.view(B, T, 3, self.num_heads, d)
        q, k, v = packed.unbind(dim=2)

        if self.context is not None:
            # Ring buffer path (sliding window attention)
            return self._forward_with_context(q, k, v, state, B, T, query.device)
        else:
            # Append path (full-sequence attention)
            return self._forward_without_context(q, k, v, state, B, T)

    def _forward_with_context(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: dict,
        B: int,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Forward pass with sliding window attention (ring buffer KV cache)."""
        offset = state["offset"]

        # q, k, v are [B, T, H, D]
        # RoPE requires [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.rope(q, k, offset)

        # Buffer storage expects [B, H, T, D] (matches cache layout of [B, H, Cap, D])
        # So we pass k, v directly (already [B, H, T, D])
        # v needs to be [B, H, T, D] too.

        # Complete KV cache
        kv_result = _complete_ring_buffer(state["cache"], state["end_offset"], k, v)

        # k_out, v_out are [B, H, Cap, D] - ready for attention
        k_out, v_out = kv_result.keys, kv_result.values
        pos_k = kv_result.positions

        # Build attention bias for sliding window
        pos_k = pos_k[:, None]
        pos_q = offset.view(-1, 1, 1) + torch.arange(T, device=device, dtype=torch.long).view(-1, 1)
        delta = pos_q - pos_k
        attn_bias = (pos_k >= 0) & (delta >= 0) & (delta < self.context)
        attn_bias = attn_bias[:, None]

        x = F.scaled_dot_product_attention(q, k_out, v_out, attn_bias, dropout_p=0.0)
        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_proj(x)
        return x

    def _forward_without_context(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: Dict[str, torch.Tensor],
        B: int,
        T: int,
    ) -> torch.Tensor:
        """Forward pass with full-sequence attention (append KV cache)."""
        current_end = state["current_end"].shape[0]

        # q, k, v are [B, T, H, D]
        device = q.device

        # RoPE expects [B, H, T, D]
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)

        # Create scalar tensor for offset to satisfy JIT/rope unannotated args
        offset_tensor = q.new_full((), current_end)
        q, k = self.rope(q, k, offset=offset_tensor)

        # Back to [B, T, H, D] for storage
        k_store = k.transpose(1, 2)

        # NOTE: v is not rotated.

        # Check if cache needs resizing
        # cache shape: [2, B, Cap, H, D]
        cache = state["cache"]
        added_length = k_store.shape[1]
        needed_capacity = current_end + added_length
        current_capacity = cache.shape[2]

        if needed_capacity > current_capacity:
            # Resize cache (double capacity or fit needed)
            new_capacity = max(needed_capacity, current_capacity * 2)
            # Match shape except capacity
            new_shape = list(cache.shape)
            new_shape[2] = new_capacity

            new_cache = cache.new_full(new_shape, float("NaN"))
            # Copy existing content
            new_cache[:, :, :current_capacity, :, :] = cache
            state["cache"] = new_cache

        # Append to buffer
        # k_buf output is full sequence [B, T_full, H, D]
        # Inline _complete_append_buffer
        added_len = k_store.shape[1]
        cache = state["cache"]
        cache[0, :, current_end : current_end + added_len] = k_store
        cache[1, :, current_end : current_end + added_len] = v

        k_full = cache[0, :, : current_end + added_len]
        v_full = cache[1, :, : current_end + added_len]

        # Attention requires [B, H, T, D]
        k = k_full.transpose(1, 2)
        v = v_full.transpose(1, 2)

        # Build causal mask
        # T is current seq len. T_full is T + current_end
        mask_shape = (T, T + current_end)

        # Inline _materialize_causal_mask
        shift = current_end
        # Use q.new_full to avoid global torch
        mask_tensor = q.new_full(mask_shape, 1.0)
        attn_mask = mask_tensor.tril(diagonal=shift)
        attn_mask = attn_mask.log()

        # Manual attention to avoid global F.scaled_dot_product_attention
        # q: [B, H, T, D], k: [B, H, S, D], v: [B, H, S, D]
        d_head = q.shape[-1]
        # scale = 1 / sqrt(d_head)
        scale = d_head**-0.5

        # attn_weights: [B, H, T, S]
        attn_weights = q.matmul(k.transpose(-2, -1)) * scale

        # attn_mask: [T, S] broadcast to [B, H, T, S]
        attn_weights = attn_weights + attn_mask

        attn_weights = attn_weights.softmax(dim=-1)

        x = attn_weights.matmul(v)  # [B, H, T, D]
        x = x.transpose(1, 2)

        # Reshape and project
        b, t, h, d = x.shape
        x = x.reshape(b, t, h * d)
        x = self.out_proj(x)
        return x
