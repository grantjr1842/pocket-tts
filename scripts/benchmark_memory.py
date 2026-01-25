#!/usr/bin/env python
"""
Memory benchmark script for Pocket TTS.

This script benchmarks memory usage across different scenarios and batch sizes.
"""

import gc
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import torch

# Setup path to import pocket_tts
sys.path.insert(0, str(Path(__file__).parent.parent))

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.utils.config import load_config
from pocket_tts.utils.utils import download_if_necessary

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    scenario: str
    batch_size: int
    text_length: int
    generation_time: float
    peak_memory_mb: float
    final_memory_mb: float
    memory_growth_mb: float
    reference_cycles: int


class MemoryBenchmark:
    """Memory benchmarking suite for Pocket TTS."""

    def __init__(self, model: TTSModel):
        """
        Initialize benchmark suite.

        Args:
            model: TTS model instance
        """
        self.model = model
        self.results: List[BenchmarkResult] = []

    def run_benchmark(
        self,
        scenario: str,
        text: str,
        batch_size: int,
    ) -> BenchmarkResult:
        """
        Run a single benchmark.

        Args:
            scenario: Name/description of the benchmark scenario
            text: Text to generate
            batch_size: Batch size to use

        Returns:
            BenchmarkResult with metrics
        """
        logger.info(f"\n=== Benchmark: {scenario} ===")
        logger.info(f"Text length: {len(text)} chars")
        logger.info(f"Batch size: {batch_size}")

        # Clear memory before benchmark
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Get initial memory
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / 1024**2
        else:
            initial_memory = 0

        # Generate audio
        start_time = time.time()
        self.model.generate(text=text, batch_size=batch_size)
        generation_time = time.time() - start_time

        # Get peak memory
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            final_memory = torch.cuda.memory_allocated() / 1024**2
        else:
            peak_memory = 0
            final_memory = 0

        memory_growth = final_memory - initial_memory

        # Check for reference cycles
        reference_cycles = gc.collect()

        result = BenchmarkResult(
            scenario=scenario,
            batch_size=batch_size,
            text_length=len(text),
            generation_time=generation_time,
            peak_memory_mb=peak_memory,
            final_memory_mb=final_memory,
            memory_growth_mb=memory_growth,
            reference_cycles=reference_cycles,
        )

        # Log results
        logger.info(f"Generation time: {generation_time:.2f}s")
        if torch.cuda.is_available():
            logger.info(f"Peak memory: {peak_memory:.2f} MB")
            logger.info(f"Memory growth: {memory_growth:.2f} MB")
        logger.info(f"Reference cycles: {reference_cycles}")

        self.results.append(result)
        return result

    def run_batch_size_comparison(
        self,
        text: str,
        batch_sizes: List[int],
        scenario_name: str = "batch_comparison",
    ) -> List[BenchmarkResult]:
        """
        Compare memory usage across different batch sizes.

        Args:
            text: Text to generate
            batch_sizes: List of batch sizes to test
            scenario_name: Name for this comparison

        Returns:
            List of BenchmarkResults
        """
        logger.info(f"\n### Batch Size Comparison: {scenario_name} ###")

        results = []
        for batch_size in batch_sizes:
            result = self.run_benchmark(
                scenario=f"{scenario_name}_batch_{batch_size}",
                text=text,
                batch_size=batch_size,
            )
            results.append(result)

        # Summary
        logger.info(f"\n### Summary for {scenario_name} ###")
        for result in results:
            logger.info(
                f"Batch {result.batch_size}: "
                f"{result.generation_time:.2f}s, "
                f"{result.peak_memory_mb:.2f} MB peak"
            )

        return results

    def run_text_length_comparison(
        self,
        texts: List[tuple[str, str]],
        batch_size: int,
        scenario_name: str = "text_length_comparison",
    ) -> List[BenchmarkResult]:
        """
        Compare memory usage across different text lengths.

        Args:
            texts: List of (name, text) tuples
            batch_size: Batch size to use
            scenario_name: Name for this comparison

        Returns:
            List of BenchmarkResults
        """
        logger.info(f"\n### Text Length Comparison: {scenario_name} ###")

        results = []
        for name, text in texts:
            result = self.run_benchmark(
                scenario=f"{scenario_name}_{name}",
                text=text,
                batch_size=batch_size,
            )
            results.append(result)

        # Summary
        logger.info(f"\n### Summary for {scenario_name} ###")
        for result in results:
            logger.info(
                f"{result.scenario}: "
                f"{result.text_length} chars, "
                f"{result.generation_time:.2f}s, "
                f"{result.peak_memory_mb:.2f} MB peak"
            )

        return results

    def test_memory_leak(
        self, text: str, batch_size: int, iterations: int = 10
    ) -> bool:
        """
        Test for memory leaks over multiple generations.

        Args:
            text: Text to generate
            batch_size: Batch size to use
            iterations: Number of iterations to run

        Returns:
            True if leak detected (memory growth > 10%), False otherwise
        """
        logger.info(f"\n### Memory Leak Test ({iterations} iterations) ###")

        memory_readings = []

        for i in range(iterations):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Generate
            self.model.generate(text=text, batch_size=batch_size)

            # Record memory
            gc.collect()
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / 1024**2
            else:
                memory_mb = 0

            memory_readings.append(memory_mb)
            logger.info(f"Iteration {i + 1}/{iterations}: {memory_mb:.2f} MB")

        # Check for leak
        if len(memory_readings) < 2:
            return False

        initial = memory_readings[0]
        final = memory_readings[-1]
        growth = final - initial
        growth_percent = (growth / initial * 100) if initial > 0 else 0

        logger.info(f"Initial memory: {initial:.2f} MB")
        logger.info(f"Final memory: {final:.2f} MB")
        logger.info(f"Growth: {growth:.2f} MB ({growth_percent:.1f}%)")

        # Consider it a leak if memory grows more than 10%
        has_leak = growth_percent > 10

        if has_leak:
            logger.warning(f"Memory leak detected! Growth: {growth_percent:.1f}%")
        else:
            logger.info(
                f"No significant memory leak detected (growth: {growth_percent:.1f}%)"
            )

        return has_leak

    def save_results(self, output_path: str | Path = "memory_benchmark_results.json"):
        """
        Save benchmark results to JSON file.

        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)

        # Convert to serializable format
        results_dict = {
            "timestamp": time.time(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "results": [asdict(r) for r in self.results],
        }

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def main():
    """Main entry point for memory benchmarking."""
    logger.info("Starting memory benchmark for Pocket TTS")

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

    # Create benchmark suite
    benchmark = MemoryBenchmark(model)

    # Test scenarios
    # 1. Batch size comparison
    benchmark.run_batch_size_comparison(
        text="This is a test of the memory benchmarking system.",
        batch_sizes=[1, 2, 4],
        scenario_name="short_text",
    )

    # 2. Text length comparison
    benchmark.run_text_length_comparison(
        texts=[
            ("short", "Short text."),
            (
                "medium",
                "This is a medium length text that contains more words to test memory usage.",
            ),
            (
                "long",
                "This is a much longer text that should require more memory to process. "
                * 5,
            ),
        ],
        batch_size=1,
        scenario_name="text_length",
    )

    # 3. Memory leak test
    benchmark.test_memory_leak(
        text="Memory leak test text.",
        batch_size=1,
        iterations=10,
    )

    # Save results
    benchmark.save_results()

    logger.info("\nMemory benchmark complete")


if __name__ == "__main__":
    main()
