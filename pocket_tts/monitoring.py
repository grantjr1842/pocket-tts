import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger()


class PerformanceMonitor:
    """
    Monitors performance metrics for operations.
    """

    def __init__(self):
        self._metrics: Dict[str, Any] = {}

    @contextmanager
    def measure(self, operation: str, **kwargs):
        """
        Measure the duration of an operation.

        Args:
            operation: Name of the operation to measure
            **kwargs: Additional context to log
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self._metrics[operation] = duration

            # Log the metric immediately
            logger.info(
                "performance_metric",
                operation=operation,
                duration_seconds=duration,
                **kwargs,
            )

    def get_metric(self, operation: str) -> Optional[float]:
        """Get the last recorded duration for an operation."""
        return self._metrics.get(operation)

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all recorded metrics."""
        return self._metrics.copy()


# Global monitor instance
monitor = PerformanceMonitor()
