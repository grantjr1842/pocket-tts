import logging
import queue as queue_module
import time
from typing import Dict, Iterator, Optional

import torch

logger = logging.getLogger(__name__)


class AdaptiveChunker:
    """
    Aggregates small audio chunks into larger ones for more efficient streaming.

    Args:
        min_chunk_size: Minimum number of samples per chunk before yielding.
        max_chunk_size: Maximum number of samples per chunk.
        sample_rate: Audio sample rate.
    """

    def __init__(
        self,
        min_chunk_size: int = 2400,
        max_chunk_size: int = 9600,
        sample_rate: int = 24000,
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.sample_rate = sample_rate
        self.buffer = []
        self.current_size = 0
        self.first_chunk_sent = False

    def add(self, chunk: torch.Tensor) -> Optional[torch.Tensor]:
        """Add a chunk and return an aggregated chunk if enough data is accumulated."""
        self.buffer.append(chunk)
        self.current_size += chunk.shape[-1]

        # Lower TTFC by sending the very first model output immediately
        if not self.first_chunk_sent:
            self.first_chunk_sent = True
            return self.flush()

        if self.current_size >= self.min_chunk_size:
            return self.flush()
        return None

    def flush(self) -> Optional[torch.Tensor]:
        """Flush the current buffer and return the aggregated chunk."""
        if not self.buffer:
            return None

        result = torch.cat(self.buffer, dim=-1)
        self.buffer = []
        self.current_size = 0
        return result


def stream_with_adaptive_chunking(
    iterator: Iterator[torch.Tensor],
    min_chunk_size: int = 2400,
    sample_rate: int = 24000,
) -> Iterator[torch.Tensor]:
    """Wrapper for an iterator to provide adaptive chunking."""
    chunker = AdaptiveChunker(min_chunk_size=min_chunk_size, sample_rate=sample_rate)
    for chunk in iterator:
        aggregated = chunker.add(chunk)
        if aggregated is not None:
            yield aggregated

    final = chunker.flush()
    if final is not None:
        yield final


class StreamingMetrics:
    """Tracks streaming performance metrics."""

    def __init__(self):
        self.start_time = None
        self.first_chunk_time = None
        self.last_chunk_time = None
        self.total_samples = 0
        self.chunk_count = 0

    def start(self):
        self.start_time = time.perf_counter()

    def on_chunk(self, chunk: torch.Tensor):
        now = time.perf_counter()
        if self.first_chunk_time is None:
            self.first_chunk_time = now
        self.last_chunk_time = now
        self.total_samples += chunk.shape[-1]
        self.chunk_count += 1

    def get_report(self, sample_rate: int) -> dict:
        if self.start_time is None or self.last_chunk_time is None:
            return {}

        total_time = self.last_chunk_time - self.start_time
        audio_duration = self.total_samples / sample_rate
        rtf = audio_duration / total_time if total_time > 0 else 0
        ttfc = (self.first_chunk_time - self.start_time) if self.first_chunk_time else 0

        return {
            "total_time_seconds": total_time,
            "audio_duration_seconds": audio_duration,
            "real_time_factor": rtf,
            "time_to_first_chunk_ms": ttfc * 1000,
            "chunk_count": self.chunk_count,
            "total_samples": self.total_samples,
        }


class QueueDepthMonitor:
    """
    Monitors queue depths during streaming generation.

    Tracks min/max/average queue sizes and provides warnings
    when queues grow beyond safe thresholds.

    NOT thread-safe - each generation should use its own instance.
    """

    def __init__(
        self,
        latents_queue: queue_module.Queue,
        result_queue: queue_module.Queue,
        warning_threshold: int = 10,
        critical_threshold: int = 20,
        log_interval_seconds: float = 1.0,
    ):
        """
        Args:
            latents_queue: Queue for generated latents
            result_queue: Queue for decoded audio chunks
            warning_threshold: Log warning if queue exceeds this size
            critical_threshold: Log critical if queue exceeds this size
            log_interval_seconds: How often to log queue statistics
        """
        self.latents_queue = latents_queue
        self.result_queue = result_queue
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.log_interval_seconds = log_interval_seconds

        # Statistics tracking
        self.start_time = None
        self.last_log_time = None
        self.latents_depth_samples = []
        self.result_depth_samples = []
        self.max_latents_depth = 0
        self.max_result_depth = 0
        self.warning_triggered = False
        self.critical_triggered = False

    def start(self):
        """Start monitoring."""
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        logger.info("QueueDepthMonitor started")

    def update(self):
        """Record current queue depths and log if needed."""
        if self.start_time is None:
            return

        now = time.perf_counter()
        latents_depth = self.latents_queue.qsize()
        result_depth = self.result_queue.qsize()

        # Track statistics
        self.latents_depth_samples.append(latents_depth)
        self.result_depth_samples.append(result_depth)
        self.max_latents_depth = max(self.max_latents_depth, latents_depth)
        self.max_result_depth = max(self.max_result_depth, result_depth)

        # Check thresholds
        if latents_depth >= self.critical_threshold or result_depth >= self.critical_threshold:
            self.critical_triggered = True
            logger.critical(
                "QueueDepthMonitor: CRITICAL - latents=%d, result=%d (threshold=%d)",
                latents_depth,
                result_depth,
                self.critical_threshold,
            )
        elif latents_depth >= self.warning_threshold or result_depth >= self.warning_threshold:
            self.warning_triggered = True
            logger.warning(
                "QueueDepthMonitor: WARNING - latents=%d, result=%d (threshold=%d)",
                latents_depth,
                result_depth,
                self.warning_threshold,
            )

        # Periodic logging
        if now - self.last_log_time >= self.log_interval_seconds:
            self._log_statistics(now)
            self.last_log_time = now

    def _log_statistics(self, now: float):
        """Log periodic queue statistics."""
        elapsed = now - self.start_time
        avg_latents = (
            sum(self.latents_depth_samples) / len(self.latents_depth_samples)
            if self.latents_depth_samples
            else 0
        )
        avg_result = (
            sum(self.result_depth_samples) / len(self.result_depth_samples)
            if self.result_depth_samples
            else 0
        )

        logger.debug(
            "QueueDepthMonitor: elapsed=%.2fs, "
            "latents: cur=%d avg=%.1f max=%d, "
            "result: cur=%d avg=%.1f max=%d",
            elapsed,
            self.latents_queue.qsize(),
            avg_latents,
            self.max_latents_depth,
            self.result_queue.qsize(),
            avg_result,
            self.max_result_depth,
        )

    def get_report(self) -> Dict:
        """Get final queue statistics report."""
        return {
            "max_latents_depth": self.max_latents_depth,
            "max_result_depth": self.max_result_depth,
            "avg_latents_depth": (
                sum(self.latents_depth_samples) / len(self.latents_depth_samples)
                if self.latents_depth_samples
                else 0
            ),
            "avg_result_depth": (
                sum(self.result_depth_samples) / len(self.result_depth_samples)
                if self.result_depth_samples
                else 0
            ),
            "warning_triggered": self.warning_triggered,
            "critical_triggered": self.critical_triggered,
            "total_samples": len(self.latents_depth_samples),
        }


class AdaptiveChunkerV2(AdaptiveChunker):
    """
    Enhanced AdaptiveChunker with text-length-aware chunk sizing.

    Extends AdaptiveChunker to dynamically adjust chunk sizes based on
    input text length, optimizing for either low TTFC (short texts) or
    high throughput (long texts).

    Backward compatible: Falls back to parent class behavior if
    text_length info not provided or adaptive is disabled.
    """

    def __init__(
        self,
        min_chunk_size: int = 2400,
        max_chunk_size: int = 9600,
        sample_rate: int = 24000,
        text_length_chars: int = 0,
        text_length_words: int = 0,
        enable_adaptive: bool = True,
    ):
        """
        Initialize AdaptiveChunkerV2.

        Args:
            min_chunk_size: Base minimum number of samples per chunk.
            max_chunk_size: Base maximum number of samples per chunk.
            sample_rate: Audio sample rate.
            text_length_chars: Input text character count (for adaptation).
            text_length_words: Input text word count (for adaptation).
            enable_adaptive: If True and text_length_words > 0, enables
                adaptive chunk sizing. If False, uses fixed sizes.
        """
        # First initialize parent class with base chunk sizes
        super().__init__(min_chunk_size, max_chunk_size, sample_rate)

        # Apply adaptive sizing if enabled and text length is provided
        if enable_adaptive and text_length_words > 0:
            (
                self.min_chunk_size,
                self.max_chunk_size,
            ) = self._calculate_adaptive_sizes(text_length_words, sample_rate)

            logger.info(
                "AdaptiveChunkerV2: text_words=%d, min=%d samples (%.1fms), "
                "max=%d samples (%.1fms)",
                text_length_words,
                self.min_chunk_size,
                (self.min_chunk_size / sample_rate) * 1000,
                self.max_chunk_size,
                (self.max_chunk_size / sample_rate) * 1000,
            )

    def _calculate_adaptive_sizes(
        self, text_length_words: int, sample_rate: int
    ) -> tuple[int, int]:
        """
        Calculate adaptive chunk sizes based on text length.

        Strategy:
        - Short texts (< 20 words): Smaller chunks for ultra-low TTFC
        - Medium texts (20-99 words): Moderate chunks for balance
        - Long texts (100+ words): Larger chunks for throughput

        Args:
            text_length_words: Number of words in input text.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Tuple of (min_chunk_size_samples, max_chunk_size_samples).
        """
        # Scale factor: 0.5x for short texts, 2.0x for very long texts
        # Clamped to ensure reasonable chunk sizes
        scale_factor = min(2.0, max(0.5, text_length_words / 50.0))

        # Base chunk sizes in milliseconds
        base_min_ms = 100
        base_max_ms = 400

        # Apply scaling
        min_chunk_ms = base_min_ms * scale_factor
        max_chunk_ms = base_max_ms * scale_factor

        # Convert to samples
        min_samples = int((min_chunk_ms / 1000.0) * sample_rate)
        max_samples = int((max_chunk_ms / 1000.0) * sample_rate)

        return min_samples, max_samples


def stream_with_adaptive_chunking_v2(
    iterator: Iterator[torch.Tensor],
    text_length_words: int = 0,
    min_chunk_size: int = 2400,
    sample_rate: int = 24000,
    enable_adaptive: bool = True,
) -> Iterator[torch.Tensor]:
    """
    Wrapper for adaptive chunking with text-length awareness.

    Args:
        iterator: Audio chunk iterator from the model.
        text_length_words: Input text word count (enables adaptation).
        min_chunk_size: Base minimum chunk size in samples.
        sample_rate: Audio sample rate in Hz.
        enable_adaptive: If False, uses fixed sizes (backward compatible).

    Yields:
        torch.Tensor: Aggregated audio chunks.
    """
    chunker = AdaptiveChunkerV2(
        min_chunk_size=min_chunk_size,
        max_chunk_size=min_chunk_size * 4,  # max is 4x min
        sample_rate=sample_rate,
        text_length_words=text_length_words,
        enable_adaptive=enable_adaptive,
    )

    for chunk in iterator:
        aggregated = chunker.add(chunk)
        if aggregated is not None:
            yield aggregated

    final = chunker.flush()
    if final is not None:
        yield final
