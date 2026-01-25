import logging
import time
from typing import Iterator, Optional

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
