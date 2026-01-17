"""Backward compatibility module.

This module re-exports StreamingMultiheadAttention from the new unified
streaming_attention module for backward compatibility. New code should
import directly from pocket_tts.modules.streaming_attention.
"""

# Re-export for backward compatibility
from pocket_tts.modules.streaming_attention import (
    StreamingMultiheadAttention,
    _complete_append_buffer as complete_kv,
    _materialize_causal_mask,
)

__all__ = ["StreamingMultiheadAttention", "complete_kv", "_materialize_causal_mask"]
