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
from pocket_tts.utils.export_wrapper import FlowLMExportWrapper

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

    # Use FlowLMExportWrapper to handle non-scriptable components and state dictionaries
    try:
    try:
        # Instantiate Wrapper
        wrapper = FlowLMExportWrapper(model.flow_lm)
        wrapper.eval()

        # Scripting
        logger.info("Scripting FlowLM using ExportWrapper...")
        scripted = torch.jit.script(wrapper)
        logger.info("Successfully scripted FlowLM model")

        torch.jit.save(scripted, output_path)
        logger.info("Saved TorchScript FlowLM to %s", output_path)

    except Exception as e:
        import traceback

        logger.error("TorchScript export failed: %s\n%s", e, traceback.format_exc())
        logger.warning(
            "Skipping FlowLM export due to error: Could not export FlowLM to TorchScript: %s", e
        )
        return None

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
        # Dummy input for decoder - latent tensor and state
        # SEANetDecoder expects (x, model_state)
        # Passing empty dict for model_state should        # StatefulModule expects dict[str, dict[str, Tensor]].
        # Also need to run init_state to populate _module_absolute_name in StatefulModules
        from pocket_tts.modules.stateful_module import init_states

        # Initialize states using the helper (which recurses and sets names)
        # We use a dummy batch size of 1 and dummy seq len 10
        dummy_state = init_states(decoder, 1, 10)

        with torch.no_grad():
            # Use correct input dimension from decoder itself
            dummy_latent = torch.randn((1, decoder.dimension, 10))
            scripted = torch.jit.trace(
                decoder, (dummy_latent, dummy_state), strict=False, check_trace=False
            )

        torch.jit.save(scripted, output_path)
        logger.info("Saved TorchScript Mimi decoder to %s", output_path)

    except Exception as e:
        import traceback

        logger.error("Failed to export Mimi decoder: %s\n%s", e, traceback.format_exc())
        raise

    return output_path


def export_to_torchscript(
    model: TTSModel, output_dir: str | Path, components: str = "all"
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
