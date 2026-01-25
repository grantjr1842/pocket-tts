"""
Batch size optimization utilities for Pocket TTS.

This module provides utilities for optimizing batch sizes based on
available memory and text length.
"""

import gc
import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pocket_tts.models.tts_model import TTSModel

logger = logging.getLogger(__name__)

# Default batch sizes for different scenarios
DEFAULT_BATCH_SIZE = 1
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 8

# Memory thresholds (in MB)
MEMORY_WARNING_THRESHOLD = 500  # Warn if memory usage exceeds this
MEMORY_ERROR_THRESHOLD = 1000  # Error if memory usage exceeds this


def get_available_memory_mb() -> float:
    """Get available system memory in MB."""
    if torch.cuda.is_available():
        # For CUDA, get free GPU memory
        device = torch.device("cuda")
        free_memory = torch.cuda.mem_get_info(device)[0] / 1024**2
        return free_memory
    else:
        # For CPU, we can't easily get accurate memory info without psutil
        # Return a conservative estimate
        return 2048.0  # Assume 2GB available


def estimate_memory_for_batch_size(
    model: "TTSModel",
    text: str,
    batch_size: int,
) -> float:
    """
    Estimate memory required for a given batch size.

    This is a heuristic estimation based on:
    - Model parameters
    - Text length
    - Batch size

    Args:
        model: TTS model instance
        text: Text to generate
        batch_size: Batch size to estimate for

    Returns:
        Estimated memory usage in MB
    """
    # Base model memory (parameter size)
    param_memory = (
        sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    )

    # Activation memory (rough estimate based on sequence length)
    text_length = len(text)
    activation_memory = text_length * batch_size * 0.1  # Rough heuristic

    # Buffer memory for intermediate results
    buffer_memory = param_memory * 0.5

    total_memory = param_memory + activation_memory + buffer_memory

    return total_memory


def find_optimal_batch_size(
    model: "TTSModel",
    text: str,
    max_batch_size: int = MAX_BATCH_SIZE,
    min_batch_size: int = MIN_BATCH_SIZE,
    safety_margin: float = 0.8,
) -> int:
    """
    Find optimal batch size based on available memory.

    Uses binary search to find the largest batch size that fits in memory.

    Args:
        model: TTS model instance
        text: Text to generate
        max_batch_size: Maximum batch size to consider
        min_batch_size: Minimum batch size to consider
        safety_margin: Safety margin (0.0-1.0) to avoid OOM

    Returns:
        Optimal batch size
    """
    available_memory = get_available_memory_mb()
    target_memory = available_memory * safety_margin

    logger.info(f"Finding optimal batch size (target memory: {target_memory:.0f} MB)")

    low = min_batch_size
    high = max_batch_size
    optimal_batch_size = min_batch_size

    while low <= high:
        mid = (low + high) // 2
        estimated_memory = estimate_memory_for_batch_size(model, text, mid)

        if estimated_memory <= target_memory:
            optimal_batch_size = mid
            low = mid + 1
        else:
            high = mid - 1

    logger.info(
        f"Optimal batch size: {optimal_batch_size} (estimated memory: {estimate_memory_for_batch_size(model, text, optimal_batch_size):.0f} MB)"
    )
    return optimal_batch_size


def adaptive_batch_generation(
    model: "TTSModel",
    text: str,
    max_batch_size: int = MAX_BATCH_SIZE,
    initial_batch_size: int | None = None,
) -> list:
    """
    Generate audio with adaptive batch size based on memory constraints.

    Monitors memory usage during generation and adjusts batch size if needed.

    Args:
        model: TTS model instance
        text: Text to generate
        max_batch_size: Maximum batch size to try
        initial_batch_size: Initial batch size (None to auto-detect)

    Returns:
        List of generated audio chunks
    """
    if initial_batch_size is None:
        initial_batch_size = find_optimal_batch_size(model, text, max_batch_size)

    logger.info(
        f"Starting adaptive batch generation with batch_size={initial_batch_size}"
    )

    audio_chunks = []
    current_batch_size = initial_batch_size

    try:
        # Attempt generation with current batch size
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        audio = model.generate(text=text, batch_size=current_batch_size)
        audio_chunks.append(audio)

        # Check memory usage after generation
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**2
            logger.info(f"Memory used after generation: {memory_used:.0f} MB")

            # Reduce batch size for next iteration if memory is high
            if (
                memory_used > MEMORY_WARNING_THRESHOLD
                and current_batch_size > MIN_BATCH_SIZE
            ):
                new_batch_size = max(MIN_BATCH_SIZE, current_batch_size // 2)
                logger.warning(
                    f"High memory usage ({memory_used:.0f} MB), reducing batch size from {current_batch_size} to {new_batch_size}"
                )
                current_batch_size = new_batch_size

        return audio_chunks

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(
                f"OOM error with batch_size={current_batch_size}, trying smaller batch"
            )
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Retry with smaller batch size
            if current_batch_size > MIN_BATCH_SIZE:
                new_batch_size = max(MIN_BATCH_SIZE, current_batch_size // 2)
                return adaptive_batch_generation(
                    model, text, max_batch_size, new_batch_size
                )
            else:
                raise RuntimeError(
                    f"Cannot complete generation even with minimum batch size: {e}"
                )
        else:
            raise


def get_recommended_batch_size(
    text_length: int, available_memory_mb: float | None = None
) -> int:
    """
    Get recommended batch size based on text length and available memory.

    Args:
        text_length: Length of text to generate
        available_memory_mb: Available memory in MB (None to use default)

    Returns:
        Recommended batch size
    """
    if available_memory_mb is None:
        available_memory_mb = 2048.0  # Default to 2GB

    # Simple heuristic: shorter text can use larger batches
    if text_length < 50:
        return min(MAX_BATCH_SIZE, 4)
    elif text_length < 200:
        return min(MAX_BATCH_SIZE, 2)
    else:
        return DEFAULT_BATCH_SIZE
