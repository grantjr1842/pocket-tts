"""
Profiling utilities for measuring model loading performance.
"""

import logging
import time
from contextlib import contextmanager
from typing import Generator, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LoadingProfiler:
    """Profile model loading time breakdown."""

    def __init__(self) -> None:
        """Initialize profiler."""
        self.timings: dict[str, float] = {}
        self._start_time: float | None = None
        self._stage_start: float | None = None

    def start(self) -> None:
        """Start profiling."""
        self._start_time = time.time()
        logger.info("Model loading profiling started")

    def record_stage(self, stage_name: str) -> None:
        """Record completion of a loading stage.

        Args:
            stage_name: Name of the stage that just completed
        """
        if self._stage_start is not None:
            elapsed = time.time() - self._stage_start
            self.timings[stage_name] = elapsed
            logger.info(f"  {stage_name}: {elapsed:.2f}s")

        self._stage_start = time.time()

    def finish(self) -> dict[str, float]:
        """Finish profiling and return timings.

        Returns:
            Dictionary mapping stage names to elapsed times
        """
        if self._start_time is None:
            return {}

        total_time = time.time() - self._start_time
        self.timings["total"] = total_time

        logger.info(f"Total model loading time: {total_time:.2f}s")
        logger.info("Breakdown:")
        for stage, elapsed in self.timings.items():
            if stage != "total":
                pct = (elapsed / total_time * 100) if total_time > 0 else 0
                logger.info(f"  {stage}: {elapsed:.2f}s ({pct:.1f}%)")

        return self.timings


@contextmanager
def profile_loading(
    profiler: LoadingProfiler | None = None,
) -> Generator[LoadingProfiler, None, None]:
    """Context manager for profiling model loading.

    Args:
        profiler: Existing profiler to use, or None to create a new one

    Yields:
        LoadingProfiler instance for recording stages
    """
    if profiler is None:
        profiler = LoadingProfiler()

    profiler.start()
    try:
        yield profiler
    finally:
        profiler.finish()


def format_timings(timings: dict[str, float]) -> str:
    """Format timing measurements for display.

    Args:
        timings: Dictionary of stage names to elapsed times

    Returns:
        Formatted string showing timing breakdown
    """
    if not timings:
        return "No timing data available"

    lines = ["Model Loading Timings:"]
    total = timings.get("total", 0)

    for stage, elapsed in timings.items():
        if stage != "total":
            pct = (elapsed / total * 100) if total > 0 else 0
            lines.append(f"  {stage}: {elapsed:.2f}s ({pct:.1f}%)")

    if total:
        lines.append(f"  Total: {total:.2f}s")

    return "\n".join(lines)
