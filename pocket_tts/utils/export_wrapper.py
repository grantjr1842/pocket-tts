from typing import Dict, Tuple

import torch
import torch.nn as nn

# We need to import the classes to type hint or instantiate
from pocket_tts.models.flow_lm import FlowLMModel


class FlowLMExportWrapper(nn.Module):
    """
    Wrapper for FlowLMModel to make it TorchScript compatible.

    Removes unscriptable components (Conditioner) and flattens state dicts.
    """

    def __init__(self, original_model: FlowLMModel):
        super().__init__()
        # Copy necessary submodules/parameters
        self.input_linear = original_model.input_linear
        self.transformer = original_model.transformer
        self.out_norm = original_model.out_norm
        self.out_eos = original_model.out_eos
        self.flow_net = original_model.flow_net

        # Register buffers
        self.register_buffer("emb_std", original_model.emb_std)
        self.register_buffer("emb_mean", original_model.emb_mean)
        self.bos_emb = original_model.bos_emb

        # Attributes
        self.ldim = original_model.ldim
        self.dim = original_model.dim

    def forward(
        self,
        sequence: torch.Tensor,
        text_embeddings: torch.Tensor,
        model_state: Dict[str, Dict[str, torch.Tensor]],
        lsd_decode_steps: int,
        temp: float,
        noise_clamp: float,
        eos_threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Backbone Logic (Copied from FlowLMModel.backbone)
        # Handle NaN BOS using instance method
        mask_nan = sequence.isnan()
        sequence = self.bos_emb.where(mask_nan, sequence)

        input_ = self.input_linear(sequence)

        # Concatenate text embeddings
        # input_ = torch.cat([text_embeddings, input_], dim=1)
        # Replace torch.cat with instance-based slicing
        B, T_txt, D = text_embeddings.shape
        _, T_seq, _ = input_.shape
        total_len = T_txt + T_seq

        concat_out = input_.new_empty((B, total_len, D))
        concat_out[:, :T_txt] = text_embeddings
        concat_out[:, T_txt:] = input_
        input_ = concat_out

        # Transformer
        transformer_out = self.transformer(input_, model_state)

        if self.out_norm is not None:
            transformer_out = self.out_norm(transformer_out)

        # Remove prefix
        transformer_out = transformer_out[:, -sequence.shape[1] :]

        # Post-processing (Copied from FlowLMModel.forward)
        # transformer_out = transformer_out.to(torch.float32) -> Use .float()
        transformer_out = transformer_out.float()

        last_out = transformer_out[:, -1]
        out_eos = self.out_eos(last_out) > eos_threshold

        # Noise generation
        noise_shape = last_out.shape[:-1] + (self.ldim,)
        std = temp**0.5
        # noise = torch.empty... -> Use new_empty
        noise = last_out.new_empty(noise_shape)

        # Branching for noise clamp
        if noise_clamp < 0:
            noise.normal_(mean=0.0, std=std)
        else:
            # Note: trunc_normal_ is not standard on Tensor instance in older torch versions.
            # Fallback to normal_ for export compatibility.
            # Ideally: noise.trunc_normal_ if available.
            noise.normal_(mean=0.0, std=std)

        # LSD Decode
        current = noise

        for i in range(lsd_decode_steps):
            s = float(i) / lsd_decode_steps
            t = float(i + 1) / lsd_decode_steps

            # s_vec = s * torch.ones_like(...) -> instance new_ones
            ones = current.new_ones(current.shape[:-1] + (1,))
            s_vec = s * ones
            t_vec = t * ones

            flow_dir = self.flow_net(last_out, s_vec, t_vec, current)

            current = current + flow_dir / lsd_decode_steps

        return current, out_eos
