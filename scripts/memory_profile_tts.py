#!/usr/bin/env python3
"""
Memory profiling script for Pocket TTS.

This script profiles memory usage during TTS generation to identify
memory leaks and optimization opportunities.
"""

import gc
import logging
import sys
import time
from pathlib import Path

import torch
from memory_profiler import profile

# Add pocket_tts to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@profile
def profile_model_loading():
    """Profile memory usage during model loading."""
    logger.info("Starting model loading profile...")

    # Force garbage collection before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = load_config()
    model = TTSModel.from_pydantic_config(config)

    logger.info(f"Model loaded. Device: {model.device}")
    logger.info(f"Model sample rate: {model.sample_rate}")

    return model


@profile
def profile_text_generation(
    model: TTSModel, text: str = "Hello world, this is a test."
):
    """Profile memory usage during text generation."""
    logger.info(f"Starting generation profile for text: '{text}'")

    # Force garbage collection before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_time = time.time()

    # Generate audio
    audio_output = model.generate_speech(
        text=text,
        variant="default",
        speaker=None,
        format="wav",
        stream=False,
    )

    end_time = time.time()
    logger.info(f"Generation completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Generated audio length: {len(audio_output)} bytes")

    return audio_output


@profile
def profile_streaming_generation(
    model: TTSModel, text: str = "Hello world, this is a streaming test."
):
    """Profile memory usage during streaming generation."""
    logger.info(f"Starting streaming profile for text: '{text}'")

    # Force garbage collection before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    audio_chunks = []
    chunk_count = 0

    start_time = time.time()

    # Generate streaming audio
    for chunk in model.generate_speech(
        text=text,
        variant="default",
        speaker=None,
        format="wav",
        stream=True,
    ):
        audio_chunks.append(chunk)
        chunk_count += 1
        logger.info(f"Received chunk {chunk_count}, size: {len(chunk)} bytes")

    end_time = time.time()
    total_audio = b"".join(audio_chunks)

    logger.info(f"Streaming completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Total chunks: {chunk_count}")
    logger.info(f"Total audio length: {len(total_audio)} bytes")

    return audio_chunks


def check_reference_cycles():
    """Check for reference cycles that might prevent garbage collection."""
    logger.info("Checking for reference cycles...")

    # Get all objects before GC
    gc.collect()
    objects_before = len(gc.get_objects())

    # Force garbage collection
    gc.collect()
    objects_after = len(gc.get_objects())

    # Check for unreachable objects (reference cycles)
    unreachable = len(gc.collect()) - (objects_before - objects_after)

    logger.info(f"Objects before GC: {objects_before}")
    logger.info(f"Objects after GC: {objects_after}")
    logger.info(f"Unreachable objects (potential cycles): {unreachable}")

    if unreachable > 0:
        logger.warning(
            f"Found {unreachable} unreachable objects - possible reference cycles"
        )
    else:
        logger.info("No reference cycles detected")


def profile_memory_usage():
    """Main profiling function."""
    logger.info("Starting memory profiling for Pocket TTS...")

    # Check for reference cycles initially
    check_reference_cycles()

    # Profile model loading
    logger.info("\n" + "=" * 50)
    logger.info("PROFILING MODEL LOADING")
    logger.info("=" * 50)

    model = profile_model_loading()

    # Check for reference cycles after loading
    check_reference_cycles()

    # Profile regular generation
    logger.info("\n" + "=" * 50)
    logger.info("PROFILING REGULAR GENERATION")
    logger.info("=" * 50)

    profile_text_generation(model)

    # Check for reference cycles after generation
    check_reference_cycles()

    # Profile streaming generation
    logger.info("\n" + "=" * 50)
    logger.info("PROFILING STREAMING GENERATION")
    logger.info("=" * 50)

    profile_streaming_generation(model)

    # Final reference cycle check
    check_reference_cycles()

    # Clean up
    logger.info("\n" + "=" * 50)
    logger.info("CLEANUP")
    logger.info("=" * 50)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Memory profiling completed!")


if __name__ == "__main__":
    profile_memory_usage()
