"""Model export utilities for TorchScript and ONNX.

This module provides functionality to export pocket-tts model components
for faster inference using TorchScript compilation.

Note: ONNX export has limitations with streaming/stateful models, so TorchScript
is the primary export format supported.

Example:
    >>> from pocket_tts.utils.export_model import export_to_torchscript
    >>> model = TTSModel.load_model()
    >>> export_to_torchscript(model, "exported_model/")
"""

import logging
from pathlib import Path

import torch

from pocket_tts.models.tts_model import TTSModel

logger = logging.getLogger(__name__)


def export_flow_lm_to_torchscript(model: TTSModel, output_dir: Path) -> Path:
    """Export the FlowLM component to TorchScript format.

    Args:
        model: Loaded TTSModel instance.
        output_dir: Directory to save exported model files.

    Returns:
        Path to the exported TorchScript file.

    Note:
        The FlowLM is the main text-to-latent model and is the largest
        component. Compiling it provides the most performance benefit.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "flow_lm.pt"

    # Use torch.jit.script for better dynamic shape support
    # Note: torch.jit.trace would be faster but requires fixed shapes
    try:
        # First try scripting
        scripted = torch.jit.script(model.flow_lm)
        logger.info("Successfully scripted FlowLM model")
    except Exception as e:
        logger.warning("TorchScript scripting failed, falling back to tracing: %s", e)
        # Fallback to tracing with example input
        # This requires dummy inputs matching the model's expected shapes
        with torch.no_grad():
            dummy_sequence = torch.randn((1, 10, model.flow_lm.ldim), dtype=torch.float32)
            dummy_text_emb = torch.randn((1, 5, model.flow_lm.dim), dtype=torch.float32)

            # Create a valid dummy model state
            # FlowLM uses StreamingTransformer -> StreamingMultiheadAttention
            # We need to initialize state properly so tracing works.
            # We can use the model's transformer to help, or just pass a dict with correct structure if simple.
            # But StreamingTransformer.init_state() is available.
            dummy_state = model.flow_lm.transformer.init_state(1, 10) # B=1, T=10

            # Tracing doesn't handle None well for optionals sometimes, so use a float for noise_clamp
            dummy_lsd = 10
            dummy_temp = 0.8
            dummy_clamp = 10.0
            dummy_eos = 0.5

            try:
                # FlowLMModel.forward args:
                # sequence, text_embeddings, model_state, lsd_decode_steps, temp, noise_clamp, eos_threshold
                scripted = torch.jit.trace(
                    model.flow_lm,
                    example_inputs=(
                        dummy_sequence,
                        dummy_text_emb,
                        dummy_state,
                        dummy_lsd,
                        dummy_temp,
                        dummy_clamp,
                        dummy_eos
                    ),
                    strict=False,
                    check_trace=False # strict checks might fail on some dynamic ops
                )
            except Exception as trace_err:
                logger.error("TorchScript tracing also failed: %s", trace_err)
                raise RuntimeError(
                    "Could not export FlowLM to TorchScript. "
                    "The model may use dynamic control flow incompatible with export."
                ) from trace_err

    torch.jit.save(scripted, output_path)
    logger.info("Saved TorchScript FlowLM to %s", output_path)

    return output_path


def export_mimi_decoder_to_torchscript(model: TTSModel, output_dir: Path) -> Path:
    """Export the Mimi decoder component to TorchScript format.

    Args:
        model: Loaded TTSModel instance.
        output_dir: Directory to save exported model files.

    Returns:
        Path to the exported TorchScript file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "mimi_decoder.pt"

    # The decoder is more straightforward to trace
    try:
        decoder = model.mimi.decoder
        # Dummy input for decoder - latent tensor
        with torch.no_grad():
            dummy_latent = torch.randn((1, model.config.mimi.quantizer.dimension, 10))
            scripted = torch.jit.trace(decoder, (dummy_latent,), strict=False)

        torch.jit.save(scripted, output_path)
        logger.info("Saved TorchScript Mimi decoder to %s", output_path)

    except Exception as e:
        logger.error("Failed to export Mimi decoder: %s", e)
        raise

    return output_path


def export_to_torchscript(
    model: TTSModel,
    output_dir: str | Path,
    components: str = "all"
) -> dict[str, Path]:
    """Export model components to TorchScript for faster inference.

    Args:
        model: Loaded TTSModel instance.
        output_dir: Directory to save exported model files.
        components: Which components to export - "all", "flow-lm", or "mimi-decoder".

    Returns:
        Dictionary mapping component names to their exported file paths.

    Example:
        >>> model = TTSModel.load_model()
        >>> paths = export_to_torchscript(model, "./exported/")
        >>> print(paths)
        {'flow-lm': PosixPath('exported/flow_lm.pt'), 'mimi-decoder': PosixPath('exported/mimi_decoder.pt')}
    """
    output_dir = Path(output_dir)
    results = {}

    # Normalize component names
    if isinstance(components, str):
        components = components.replace("_", "-").lower()

    if components in ("all", "flow-lm"):
        try:
            results["flow-lm"] = export_flow_lm_to_torchscript(model, output_dir)
        except Exception as e:
            logger.warning("Skipping FlowLM export due to error: %s", e)

    if components in ("all", "mimi-decoder"):
        try:
            results["mimi-decoder"] = export_mimi_decoder_to_torchscript(model, output_dir)
        except Exception as e:
            logger.warning("Skipping Mimi decoder export due to error: %s", e)

    if not results:
        logger.warning("No components were successfully exported")
    else:
        logger.info("Exported %d component(s) to %s", len(results), output_dir)

    return results
