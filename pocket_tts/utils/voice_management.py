import logging
from pathlib import Path
from typing import Any, Union

import safetensors.torch
import torch
import torch.nn.functional as F

from pocket_tts.utils.utils import (
    PREDEFINED_VOICES,
    download_if_necessary,
    load_predefined_voice,
)

logger = logging.getLogger(__name__)


def load_voice_tensor(source: Union[str, Path], model: Any = None) -> torch.Tensor:
    """
    Load a voice tensor from a predefined name, a safetensors file, or an audio file.

    Args:
        source: Name of predefined voice, path to safetensors file, or path to audio file.
        model: TTSModel instance. Required only if source is an audio file that needs encoding.

    Returns:
        torch.Tensor: The voice conditioning tensor [1, T, D].
    """
    if isinstance(source, str) and source in PREDEFINED_VOICES:
        return load_predefined_voice(source)

    path = Path(source)

    # Check if it's a URL-like string that needs downloading
    if isinstance(source, str) and (
        source.startswith("http://")
        or source.startswith("https://")
        or source.startswith("hf://")
    ):
        path = download_if_necessary(source)

    if not path.exists():
        raise FileNotFoundError(f"Voice file not found: {source}")

    if path.suffix == ".safetensors":
        data = safetensors.torch.load_file(path)
        if "audio_prompt" in data:
            return data["audio_prompt"]
        else:
            # Fallback: maybe it's just the tensor saved directly? Unlikely for safetensors.
            # or maybe different key?
            raise ValueError(
                f"Safetensors file {path} does not contain 'audio_prompt' key"
            )

    # Assume audio file (wav, mp3, etc)
    if model is None:
        raise ValueError(
            f"TTSModel instance is required to encode audio file: {source}"
        )

    # Import here to avoid circular dependencies if this module is imported by others
    from pocket_tts.data.audio import audio_read
    from pocket_tts.data.audio_utils import convert_audio

    audio, sr = audio_read(path)
    # Resample to model's sample rate
    audio = convert_audio(audio, sr, model.config.mimi.sample_rate, 1)

    with torch.no_grad():
        # Use model's internal encoding logic
        prompt = model._encode_audio(audio.unsqueeze(0).to(model.device))

    return prompt.cpu()  # Return on CPU to be safe


def compute_voice_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two voice tensors.
    Uses mean pooling over the time dimension.

    Args:
        tensor1: [1, T1, D]
        tensor2: [1, T2, D]

    Returns:
        float: Cosine similarity score [-1.0, 1.0]
    """
    if tensor1.dim() != 3 or tensor2.dim() != 3:
        raise ValueError(
            f"Expected 3D tensors [1, T, D], got {tensor1.shape} and {tensor2.shape}"
        )

    # Mean pooling over time (dim 1)
    vec1 = tensor1.mean(dim=1).squeeze(0)  # [D]
    vec2 = tensor2.mean(dim=1).squeeze(0)  # [D]

    if vec1.shape != vec2.shape:
        # Should verify D matches
        raise ValueError(
            f"Voice embedding dimension mismatch: {vec1.shape} vs {vec2.shape}"
        )

    return F.cosine_similarity(vec1, vec2, dim=0).item()


def blend_voice_tensors(
    tensor1: torch.Tensor, tensor2: torch.Tensor, alpha: float = 0.5
) -> torch.Tensor:
    """
    Blend two voice tensors by interpolating and mixing.

    Args:
        tensor1: [1, T1, D]
        tensor2: [1, T2, D]
        alpha: Mixing ratio. 0.0 = 100% voice1, 1.0 = 100% voice2.

    Returns:
        torch.Tensor: Blended tensor [1, T_max, D]
    """
    if tensor1.shape[-1] != tensor2.shape[-1]:
        raise ValueError("Feature dimensions mismatch")

    alpha = max(0.0, min(1.0, alpha))

    len1 = tensor1.shape[1]
    len2 = tensor2.shape[1]
    target_len = max(len1, len2)

    # We need to interpolate the time dimension.
    # torch.nn.functional.interpolate expects [N, C, L]
    # Input is [1, T, D]. We treat D as channels (C) for 1D interpolation?
    # No, typically interpolation is over spatial/temporal dims.
    # Here T is temporal. D is features.
    # So we want [1, D, T] layout for interpolate(..., size=new_T)

    t1_p = tensor1.permute(0, 2, 1)  # [1, D, T1]
    t2_p = tensor2.permute(0, 2, 1)  # [1, D, T2]

    # Intepolate to target_len
    if len1 != target_len:
        t1_resized = F.interpolate(
            t1_p, size=target_len, mode="linear", align_corners=False
        )
    else:
        t1_resized = t1_p

    if len2 != target_len:
        t2_resized = F.interpolate(
            t2_p, size=target_len, mode="linear", align_corners=False
        )
    else:
        t2_resized = t2_p

    # Blend
    blended_p = (1.0 - alpha) * t1_resized + alpha * t2_resized

    # Permute back to [1, T, D]
    return blended_p.permute(0, 2, 1)
