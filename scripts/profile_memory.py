#!/usr/bin/env python
"""
Memory profiling script for Pocket TTS.

This script profiles memory usage during TTS generation to identify
memory leaks and optimization opportunities.
"""

import gc
import logging
import sys
import time
import tracemalloc
from pathlib import Path

import torch
from memory_profiler import profile

# Setup path to import pocket_tts
sys.path.insert(0, str(Path(__file__).parent.parent))

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.config import load_config
from pocket_tts.utils.utils import download_if_necessary

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_reference_cycles():
    """Check for reference cycles in the garbage collector."""
    logger.info("Checking for reference cycles...")
    gc.collect()
    cycles = gc.collect()
    logger.info(f"Found and collected {cycles} reference cycles")
    return cycles


def get_memory_stats():
    """Get current memory usage statistics."""
    if torch.cuda.is_available():
        cuda_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        cuda_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
    else:
        cuda_allocated = cuda_reserved = 0

    current, peak = tracemalloc.get_traced_memory()
    return {
        "tracemalloc_current_mb": current / 1024**2,
        "tracemalloc_peak_mb": peak / 1024**2,
        "cuda_allocated_mb": cuda_allocated,
        "cuda_reserved_mb": cuda_reserved,
    }


@profile
def generate_with_memory_profile(
    model: TTSModel,
    text: str,
    num_chunks: int = 3,
    batch_size: int = 1,
) -> None:
    """
    Generate audio with memory profiling.

    Args:
        model: TTS model instance
        text: Text to generate speech for
        num_chunks: Number of chunks to generate
        batch_size: Batch size for generation
    """
    logger.info(f"Generating {num_chunks} chunks with batch_size={batch_size}")
    logger.info(f"Text: {text}")

    # Initial memory check
    gc.collect()
    initial_stats = get_memory_stats()
    logger.info(f"Initial memory: {initial_stats['tracemalloc_current_mb']:.2f} MB")

    # Split text into chunks
    chunk_texts = [text] * num_chunks

    # Generate audio chunks
    audio_chunks = []
    for i, chunk_text in enumerate(chunk_texts):
        logger.info(f"\n--- Chunk {i + 1}/{num_chunks} ---")

        # Memory before generation
        gc.collect()
        before_stats = get_memory_stats()
        logger.info(f"Memory before: {before_stats['tracemalloc_current_mb']:.2f} MB")

        # Generate audio
        start_time = time.time()
        audio_chunk = model.generate(
            text=chunk_text,
            batch_size=batch_size,
        )
        generation_time = time.time() - start_time

        audio_chunks.append(audio_chunk)

        # Memory after generation
        gc.collect()
        after_stats = get_memory_stats()
        logger.info(f"Memory after: {after_stats['tracemalloc_current_mb']:.2f} MB")
        logger.info(
            f"Memory delta: {after_stats['tracemalloc_current_mb'] - before_stats['tracemalloc_current_mb']:.2f} MB"
        )
        logger.info(f"Peak memory: {after_stats['tracemalloc_peak_mb']:.2f} MB")
        logger.info(f"Generation time: {generation_time:.2f}s")

        # Check for reference cycles after each chunk
        check_reference_cycles()

    # Final memory check
    gc.collect()
    final_stats = get_memory_stats()
    logger.info("\n--- Final Statistics ---")
    logger.info(f"Final memory: {final_stats['tracemalloc_current_mb']:.2f} MB")
    logger.info(f"Peak memory: {final_stats['tracemalloc_peak_mb']:.2f} MB")
    logger.info(
        f"Total memory growth: {final_stats['tracemalloc_current_mb'] - initial_stats['tracemalloc_current_mb']:.2f} MB"
    )


def main():
    """Main entry point for memory profiling."""
    logger.info("Starting memory profiling for Pocket TTS")

    # Start tracing memory
    tracemalloc.start()

    # Load configuration
    config = load_config()
    logger.info(f"Loaded config with variant: {config.variant}")

    # Download model if necessary
    download_if_necessary(config)

    # Load model
    logger.info("Loading TTS model...")
    model = TTSModel.from_pydantic_config_with_weights(
        config=config,
        temp=0.7,
        lsd_decode_steps=4,
        noise_clamp=None,
        eos_threshold=0.5,
    )
    model.eval()
    logger.info("Model loaded")

    # Get initial memory stats
    gc.collect()
    initial_stats = get_memory_stats()
    logger.info("\n--- Initial Memory Statistics ---")
    logger.info(
        f"Tracemalloc current: {initial_stats['tracemalloc_current_mb']:.2f} MB"
    )
    logger.info(f"Tracemalloc peak: {initial_stats['tracemalloc_peak_mb']:.2f} MB")
    if torch.cuda.is_available():
        logger.info(f"CUDA allocated: {initial_stats['cuda_allocated_mb']:.2f} MB")
        logger.info(f"CUDA reserved: {initial_stats['cuda_reserved_mb']:.2f} MB")

    # Run generation with profiling
    test_text = "This is a test of memory profiling for the Pocket TTS system. We are checking for memory leaks and optimization opportunities."
    generate_with_memory_profile(model, test_text, num_chunks=3, batch_size=1)

    # Check for reference cycles at the end
    logger.info("\n--- Final Reference Cycle Check ---")
    check_reference_cycles()

    # Stop tracing memory
    tracemalloc.stop()

    logger.info("\nMemory profiling complete")


if __name__ == "__main__":
    main()
