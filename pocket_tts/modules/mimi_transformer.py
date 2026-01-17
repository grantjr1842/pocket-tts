import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Self

from pocket_tts.modules.layer_scale import LayerScale
from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.streaming_attention import StreamingMultiheadAttention
from pocket_tts.utils.config import FlowLMTransformerConfig


class StreamingTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        context: int | None,
        rope: RotaryEmbedding,
        layer_scale: float | None = None,
        attention_kind: str = "mimi",
    ):
        super().__init__()
        # Note: Both attention kinds now use the unified StreamingMultiheadAttention.
        # When context is provided, it uses sliding window attention (ring buffer KV cache).
        # When context is None, it uses full-sequence attention (append KV cache).
        if attention_kind == "mimi":
            self.self_attn = StreamingMultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, rope=rope, context=context
            )
        else:
            self.self_attn = StreamingMultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, rope=rope
            )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)

        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale)
            self.layer_scale_2 = LayerScale(d_model, layer_scale)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x_orig = x
        x = self.norm2(x)
        update = self.linear2(F.gelu(self.linear1(x)))
        return x_orig.to(update) + self.layer_scale_2(update)

    def _sa_block(self, x: torch.Tensor, model_state: dict | None) -> torch.Tensor:
        x_orig = x
        x = self.norm1(x)
        # Handle state mapping for this layer
        if model_state is not None:
             # Depending on how state is structured (nested or passed directly).
             # Usually wrapper handles extraction.
             # But here we pass 'model_state' which is the top-level dict?
             # No, StreamingTransformer passes 'model_state' to layer.
             # We should probably pass model_state[layer_id] if structured.
             # Checking StreamingMultiheadAttention.get_state:
             # It expects a dict and relies on StatefulModule.get_state logic.
             # If model_state is the WHOLE state, get_state searches by module path?
             # StatefulModule does that. So passing the root dict is fine.
             pass
        update = self.self_attn(x, model_state)
        return x_orig.to(update) + self.layer_scale_1(update)

    def forward(self, x: torch.Tensor, model_state: dict | None) -> torch.Tensor:
        x = self._sa_block(x, model_state)
        x = self._ff_block(x)
        return x


class StreamingTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        layer_scale: float | None = None,
        dim_feedforward: int | list[int] = 2048,
        context: int | None = None,
        max_period: float = 10_000.0,
        kind: str = "mimi",
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.max_period = max_period

        self.rope = RotaryEmbedding(max_period=max_period)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                StreamingTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    context=context,
                    rope=self.rope,
                    layer_scale=layer_scale,
                    attention_kind=kind,
                )
            )

    @classmethod
    def from_pydantic_config(cls, config: FlowLMTransformerConfig) -> Self:
        dim_feedforward = int(config.d_model * config.hidden_scale)
        return cls(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=dim_feedforward,
            max_period=float(config.max_period),
            kind="flow_lm",
        )

    def init_state(self, batch_size: int, sequence_length: int = 2048) -> dict:
        """Initialize the state for all layers."""
        state = {}
        # We need to rely on the underlying StatefulModule logic to name keys?
        # Or construct a hierarchical dict?
        # StatefulModule.init_state usually returns a dict with proper keys if calling on self?
        # But StreamingMultiheadAttention.init_state returns raw state dict for ITSELF.
        # It does NOT namespace it.
        #
        # If we return a dict, we need to namespace it so that
        # StreamingMultiheadAttention.get_state can find it.
        # StatefulModule keys match the module hierarchy? e.g. "transformer.layers.0.self_attn.cache"
        #
        # We can simulate this by initializing a flat dict with prefixed keys?
        # OR return nested dicts if the infrastructure supports it?
        #
        # Checking StatefulModule:
        # It likely registers state in the state_dict.
        # But here we are creating a dynamic state dict for valid inference.
        #
        # A simple approach:
        # Iterate layers, call init_state, and merge into one dict with prefix.

        for i, layer in enumerate(self.layers):
            # Use context if available, else usage default
            ctx = layer.self_attn.context
            effective_len = ctx if ctx is not None else sequence_length

            layer_state = layer.self_attn.init_state(batch_size, effective_len)

            # Key prefixing logic matching module structure
            # strict path: layers.{i}.self_attn.{key}
            for k, v in layer_state.items():
                state[f"layers.{i}.self_attn.{k}"] = v

        return state

    def forward(self, x: torch.Tensor, model_state: dict | None):
        for layer in self.layers:
            x = layer(x, model_state)
        return x


class ProjectedTransformer(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimensions: tuple[int, ...],
        d_model: int,
        num_heads: int,
        num_layers: int,
        layer_scale: float,
        context: int,
        max_period: float,
        dim_feedforward: int,
    ):
        super().__init__()
        self.transformer = StreamingTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            layer_scale=layer_scale,
            context=context,
            max_period=max_period,
            dim_feedforward=dim_feedforward,
        )
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.input_proj = None
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)

        self.output_projs = nn.ModuleList()
        for output_dimension in output_dimensions:
            if d_model == output_dimension:
                self.output_projs.append(nn.Identity())
            else:
                self.output_projs.append(nn.Linear(d_model, output_dimension, bias=False))

    def forward(self, x, model_state: dict | None):
        x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z = self.transformer(x, model_state)
        ys = []
        for output_proj in self.output_projs:
            y = output_proj(z)
            y = y.transpose(1, 2)
            ys.append(y)
        return ys
