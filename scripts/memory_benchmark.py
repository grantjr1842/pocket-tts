#!/usr/bin/env python3
"""
Memory benchmark script for Pocket TTS.

This script runs memory benchmarks to validate memory optimizations
and measure performance improvements.
"""

import gc
import logging
import sys
import time
from pathlib import Path

import torch

# Add pocket_tts to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.config import load_config
from pocket_tts.utils.memory_optimizer import (
    memory_efficient_context,
    get_memory_pool,
    get_memory_monitor,
    get_cycle_detector,
    optimize_model_memory,
    log_memory_summary,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_model_loading():
    """Benchmark model loading with and without optimizations."""
    logger.info("=== Model Loading Benchmark ===")

    # Reset stats
    get_memory_pool().clear()
    get_memory_monitor().reset_stats()

    # Load without optimizations
    logger.info("Loading model without optimizations...")
    start_time = time.time()
    config = load_config()
    model1 = TTSModel.from_pydantic_config(config)
    load_time_no_opt = time.time() - start_time

    # Measure memory
    memory_no_opt = get_memory_monitor().get_current_memory_usage()

    # Cleanup
    del model1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load with optimizations
    logger.info("Loading model with optimizations...")
    get_memory_monitor().reset_stats()

    with memory_efficient_context(memory_limit_mb=1000):
        start_time = time.time()
        config = load_config()
        model2 = TTSModel.from_pydantic_config(config)
        optimize_model_memory(model2)
        load_time_opt = time.time() - start_time

    memory_opt = get_memory_monitor().get_current_memory_usage()

    # Cleanup
    del model2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Results
    logger.info(f"Load time without optimization: {load_time_no_opt:.2f}s")
    logger.info(f"Load time with optimization: {load_time_opt:.2f}s")
    logger.info(f"Memory without optimization: {memory_no_opt / (1024 * 1024):.1f}MB")
    logger.info(f"Memory with optimization: {memory_opt / (1024 * 1024):.1f}MB")

    improvement = ((memory_no_opt - memory_opt) / memory_no_opt) * 100
    logger.info(f"Memory improvement: {improvement:.1f}%")

    return {
        "load_time_no_opt": load_time_no_opt,
        "load_time_opt": load_time_opt,
        "memory_no_opt": memory_no_opt,
        "memory_opt": memory_opt,
        "memory_improvement": improvement,
    }


def benchmark_generation(
    model: TTSModel, text: str = "Hello world, this is a memory benchmark test."
):
    """Benchmark text generation with and without optimizations."""
    logger.info("=== Generation Benchmark ===")

    # Reset stats
    get_memory_pool().clear()
    get_memory_monitor().reset_stats()

    # Generate without optimizations
    logger.info("Generating without optimizations...")
    start_time = time.time()
    audio1 = model.generate_speech(
        text=text,
        variant="default",
        speaker=None,
        format="wav",
        stream=False,
    )
    gen_time_no_opt = time.time() - start_time

    memory_no_opt = get_memory_monitor().get_peak_memory_usage()

    # Cleanup
    del audio1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Generate with optimizations
    logger.info("Generating with optimizations...")
    get_memory_monitor().reset_stats()

    with memory_efficient_context(memory_limit_mb=500):
        start_time = time.time()
        audio2 = model.generate_speech(
            text=text,
            variant="default",
            speaker=None,
            format="wav",
            stream=False,
        )
        gen_time_opt = time.time() - start_time

    memory_opt = get_memory_monitor().get_peak_memory_usage()

    # Cleanup
    del audio2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Results
    logger.info(f"Generation time without optimization: {gen_time_no_opt:.2f}s")
    logger.info(f"Generation time with optimization: {gen_time_opt:.2f}s")
    logger.info(
        f"Peak memory without optimization: {memory_no_opt / (1024 * 1024):.1f}MB"
    )
    logger.info(f"Peak memory with optimization: {memory_opt / (1024 * 1024):.1f}MB")

    improvement = ((memory_no_opt - memory_opt) / memory_no_opt) * 100
    logger.info(f"Memory improvement: {improvement:.1f}%")

    return {
        "gen_time_no_opt": gen_time_no_opt,
        "gen_time_opt": gen_time_opt,
        "memory_no_opt": memory_no_opt,
        "memory_opt": memory_opt,
        "memory_improvement": improvement,
    }


def benchmark_streaming(
    model: TTSModel,
    text: str = "Hello world, this is a streaming memory benchmark test.",
):
    """Benchmark streaming generation with memory optimizations."""
    logger.info("=== Streaming Benchmark ===")

    # Reset stats
    get_memory_pool().clear()
    get_memory_monitor().reset_stats()

    # Start monitoring
    get_memory_monitor().start_monitoring()

    # Streaming generation
    logger.info("Starting streaming generation...")
    start_time = time.time()

    chunk_count = 0
    total_audio = b""

    for chunk in model.generate_speech(
        text=text,
        variant="default",
        speaker=None,
        format="wav",
        stream=True,
    ):
        total_audio += chunk
        chunk_count += 1

        # Log memory usage every 10 chunks
        if chunk_count % 10 == 0:
            current_memory = get_memory_monitor().get_current_memory_usage()
            logger.info(
                f"Chunk {chunk_count}: Memory usage {current_memory / (1024 * 1024):.1f}MB"
            )

    stream_time = time.time() - start_time
    peak_memory = get_memory_monitor().get_peak_memory_usage()

    # Stop monitoring
    get_memory_monitor().stop_monitoring()

    # Results
    logger.info(f"Streaming time: {stream_time:.2f}s")
    logger.info(f"Total chunks: {chunk_count}")
    logger.info(f"Total audio size: {len(total_audio)} bytes")
    logger.info(f"Peak memory usage: {peak_memory / (1024 * 1024):.1f}MB")

    # Check for memory alerts
    alerts = get_memory_monitor().get_memory_alerts()
    if alerts:
        logger.warning(f"Memory alerts: {alerts}")

    return {
        "stream_time": stream_time,
        "chunk_count": chunk_count,
        "total_audio_size": len(total_audio),
        "peak_memory": peak_memory,
        "alerts": alerts,
    }


def benchmark_memory_pool():
    """Benchmark memory pool effectiveness."""
    logger.info("=== Memory Pool Benchmark ===")

    pool = get_memory_pool()
    pool.clear()

    # Test tensor allocation and reuse
    shapes = [(100,), (1000,), (10000,), (100, 100), (10, 10, 10)]

    for shape in shapes:
        logger.info(f"Testing shape {shape}...")

        # First pass (allocations)
        tensors = []
        for i in range(10):
            tensor = pool.get_tensor(shape)
            tensors.append(tensor)

        # Return tensors to pool
        for tensor in tensors:
            pool.return_tensor(tensor)

        # Second pass (should reuse from pool)
        reused_tensors = []
        for i in range(10):
            tensor = pool.get_tensor(shape)
            reused_tensors.append(tensor)

        # Return tensors to pool
        for tensor in reused_tensors:
            pool.return_tensor(tensor)

    # Log pool stats
    pool.log_stats()
    stats = pool.get_stats()

    return stats


def test_reference_cycles():
    """Test reference cycle detection."""
    logger.info("=== Reference Cycle Test ===")

    detector = get_cycle_detector()
    detector.clear_registry()

    # Create some objects that might create cycles
    class TestObject:
        def __init__(self, name):
            self.name = name
            self.references = []

        def add_reference(self, other):
            self.references.append(other)

    # Create objects with circular references
    obj1 = TestObject("obj1")
    obj2 = TestObject("obj2")
    obj1.add_reference(obj2)
    obj2.add_reference(obj1)

    # Track objects
    detector.track_object(obj1)
    detector.track_object(obj2)

    logger.info(f"Tracked objects: {detector.get_tracked_count()}")

    # Delete references
    del obj1, obj2

    # Detect cycles
    cycles_detected = detector.detect_cycles()

    logger.info(f"Reference cycles detected: {cycles_detected}")
    logger.info(f"Tracked objects after cleanup: {detector.get_tracked_count()}")

    return cycles_detected


def run_memory_benchmark():
    """Run comprehensive memory benchmark."""
    logger.info("Starting comprehensive memory benchmark...")

    results = {}

    # Test reference cycles
    results["reference_cycles"] = test_reference_cycles()

    # Test memory pool
    results["memory_pool"] = benchmark_memory_pool()

    # Benchmark model loading
    results["model_loading"] = benchmark_model_loading()

    # Load model for generation tests
    logger.info("Loading model for generation benchmarks...")
    config = load_config()
    model = TTSModel.from_pydantic_config(config)
    optimize_model_memory(model)

    try:
        # Benchmark regular generation
        results["generation"] = benchmark_generation(model)

        # Benchmark streaming
        results["streaming"] = benchmark_streaming(model)

    finally:
        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final memory summary
    log_memory_summary()

    # Summary
    logger.info("=== Benchmark Summary ===")

    if "model_loading" in results:
        loading = results["model_loading"]
        logger.info(
            f"Model loading memory improvement: {loading['memory_improvement']:.1f}%"
        )

    if "generation" in results:
        generation = results["generation"]
        logger.info(
            f"Generation memory improvement: {generation['memory_improvement']:.1f}%"
        )

    if "memory_pool" in results:
        pool_stats = results["memory_pool"]
        hit_rate = (
            pool_stats["hits"] / max(pool_stats["hits"] + pool_stats["misses"], 1) * 100
        )
        logger.info(f"Memory pool hit rate: {hit_rate:.1f}%")

    logger.info("Memory benchmark completed!")

    return results


if __name__ == "__main__":
    results = run_memory_benchmark()
