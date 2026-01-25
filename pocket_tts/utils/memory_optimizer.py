"""
Memory optimization utilities for Pocket TTS.

This module provides memory optimization features including:
- Memory pooling for tensor allocations
- Batch size optimization
- Memory usage monitoring
- Reference cycle detection and cleanup
"""

import gc
import logging
import threading
import weakref
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch import nn

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryPool:
    """
    A simple memory pool for reusing tensor allocations.
    Reduces memory fragmentation and allocation overhead.
    """

    def __init__(self, max_pool_size: int = 100):
        self.max_pool_size = max_pool_size
        self._pools: Dict[Tuple[tuple, torch.dtype], List[torch.Tensor]] = {}
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "allocations": 0}

    def get_tensor(
        self, shape: tuple, dtype: torch.dtype = torch.float32, device: str = "cpu"
    ) -> torch.Tensor:
        """Get a tensor from the pool or allocate a new one."""
        key = (shape, dtype)

        with self._lock:
            pool = self._pools.get(key, [])

            if pool:
                tensor = pool.pop()
                # Move to correct device if needed
                if tensor.device.type != device:
                    tensor = tensor.to(device)
                self._stats["hits"] += 1
                logger.debug(f"Memory pool hit: {shape} {dtype}")
                return tensor
            else:
                self._stats["misses"] += 1

        # Allocate new tensor
        tensor = torch.empty(shape, dtype=dtype, device=device)
        self._stats["allocations"] += 1
        logger.debug(f"Memory pool miss: allocated new {shape} {dtype}")
        return tensor

    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return a tensor to the pool for reuse."""
        if tensor is None:
            return

        key = (tuple(tensor.shape), tensor.dtype)

        with self._lock:
            pool = self._pools.get(key, [])

            if len(pool) < self.max_pool_size:
                # Zero out the tensor for security
                tensor.zero_()
                pool.append(tensor.detach().clone())
                self._pools[key] = pool
            # If pool is full, let the tensor be garbage collected

    def clear(self) -> None:
        """Clear all pools and free memory."""
        with self._lock:
            self._pools.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Memory pool cleared")

    def get_stats(self) -> Dict[str, int]:
        """Get memory pool statistics."""
        with self._lock:
            total_tensors = sum(len(pool) for pool in self._pools.values())
            return {
                **self._stats,
                "pooled_tensors": total_tensors,
                "pool_types": len(self._pools),
            }

    def log_stats(self) -> None:
        """Log memory pool statistics."""
        stats = self.get_stats()
        hit_rate = stats["hits"] / max(stats["hits"] + stats["misses"], 1) * 100
        logger.info(
            f"Memory Pool Stats - Hit rate: {hit_rate:.1f}%, "
            f"Pooled tensors: {stats['pooled_tensors']}, "
            f"Pool types: {stats['pool_types']}"
        )


# Global memory pool instance
_memory_pool = MemoryPool()


class BatchSizeOptimizer:
    """
    Optimizes batch sizes based on available memory and content characteristics.
    """

    def __init__(self, memory_limit_mb: Optional[int] = None):
        self.memory_limit_bytes = (
            memory_limit_mb * 1024 * 1024 if memory_limit_mb else None
        )
        self._optimal_batch_sizes: Dict[str, int] = {}

    def estimate_memory_usage(
        self, batch_size: int, sequence_length: int, model_size_mb: int = 500
    ) -> int:
        """
        Estimate memory usage for a given batch size and sequence length.

        Args:
            batch_size: Number of sequences to process
            sequence_length: Length of each sequence
            model_size_mb: Approximate model size in MB

        Returns:
            Estimated memory usage in bytes
        """
        # Rough estimation: model_size + activation_memory + gradient_memory
        # Activation memory scales with batch_size * sequence_length
        activation_memory_mb = (batch_size * sequence_length * 4) / (
            1024 * 1024
        )  # 4 bytes per float32
        total_memory_mb = (
            model_size_mb + activation_memory_mb * 2
        )  # 2x for activations + gradients

        return int(total_memory_mb * 1024 * 1024)

    def get_optimal_batch_size(
        self, sequence_length: int, model_size_mb: int = 500
    ) -> int:
        """
        Get optimal batch size based on available memory and sequence length.

        Args:
            sequence_length: Length of input sequence
            model_size_mb: Approximate model size in MB

        Returns:
            Optimal batch size
        """
        cache_key = f"seq_{sequence_length}"

        if cache_key in self._optimal_batch_sizes:
            return self._optimal_batch_sizes[cache_key]

        # Get available memory
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory
            used_memory = torch.cuda.memory_allocated()
            free_memory = available_memory - used_memory
        else:
            if PSUTIL_AVAILABLE:
                free_memory = psutil.virtual_memory().available
            else:
                # Fallback: assume 4GB available if psutil not available
                free_memory = 4 * 1024 * 1024 * 1024

        # Apply memory limit if set
        if self.memory_limit_bytes:
            free_memory = min(free_memory, self.memory_limit_bytes)

        # Binary search for optimal batch size
        low, high = 1, 32  # Start with reasonable bounds

        while low < high:
            mid = (low + high + 1) // 2
            estimated_memory = self.estimate_memory_usage(
                mid, sequence_length, model_size_mb
            )

            if estimated_memory < free_memory * 0.8:  # Use 80% of available memory
                low = mid
            else:
                high = mid - 1

        optimal_batch_size = max(1, low)
        self._optimal_batch_sizes[cache_key] = optimal_batch_size

        logger.info(
            f"Optimal batch size for seq_len={sequence_length}: {optimal_batch_size}"
        )
        return optimal_batch_size


class MemoryMonitor:
    """
    Monitors memory usage and provides alerts for potential issues.
    """

    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._peak_memory_usage = 0
        self._memory_alerts: List[str] = []

    def start_monitoring(self) -> None:
        """Start memory monitoring in background thread."""
        if self._monitoring:
            return

        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("Memory monitoring stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            try:
                current_memory = self.get_current_memory_usage()
                self._peak_memory_usage = max(self._peak_memory_usage, current_memory)

                # Check for memory alerts
                if torch.cuda.is_available():
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    usage_percent = (current_memory / total_memory) * 100

                    if usage_percent > 90:
                        alert = f"High GPU memory usage: {usage_percent:.1f}%"
                        if alert not in self._memory_alerts:
                            self._memory_alerts.append(alert)
                            logger.warning(alert)

                import time

                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                break

    def get_current_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        else:
            if PSUTIL_AVAILABLE:
                return psutil.Process().memory_info().rss
            else:
                # Fallback: return 0 if psutil not available
                return 0

    def get_peak_memory_usage(self) -> int:
        """Get peak memory usage recorded."""
        return self._peak_memory_usage

    def get_memory_alerts(self) -> List[str]:
        """Get memory alerts recorded."""
        return self._memory_alerts.copy()

    def reset_stats(self) -> None:
        """Reset monitoring statistics."""
        self._peak_memory_usage = 0
        self._memory_alerts.clear()


class ReferenceCycleDetector:
    """
    Detects and helps clean up reference cycles that prevent garbage collection.
    """

    def __init__(self):
        self._object_registry: Dict[int, weakref.ref] = {}
        self._cycle_count = 0

    def track_object(self, obj: Any) -> None:
        """Track an object for cycle detection."""
        obj_id = id(obj)
        self._object_registry[obj_id] = weakref.ref(obj, self._object_deleted)

    def _object_deleted(self, weak_ref: weakref.ref) -> None:
        """Called when a tracked object is deleted."""
        # Remove from registry
        for obj_id, ref in list(self._object_registry.items()):
            if ref is weak_ref:
                del self._object_registry[obj_id]
                break

    def detect_cycles(self) -> int:
        """
        Detect reference cycles by forcing garbage collection.

        Returns:
            Number of objects involved in cycles
        """
        # Force garbage collection
        collected = gc.collect()

        if collected > 0:
            self._cycle_count += collected
            logger.warning(f"Detected {collected} objects in reference cycles")

        return collected

    def get_tracked_count(self) -> int:
        """Get number of currently tracked objects."""
        return len(self._object_registry)

    def clear_registry(self) -> None:
        """Clear the object registry."""
        self._object_registry.clear()


# Global instances
_batch_optimizer = BatchSizeOptimizer()
_memory_monitor = MemoryMonitor()
_cycle_detector = ReferenceCycleDetector()


@contextmanager
def memory_efficient_context(memory_limit_mb: Optional[int] = None):
    """
    Context manager for memory-efficient operations.

    Args:
        memory_limit_mb: Optional memory limit in MB
    """
    original_limit = _batch_optimizer.memory_limit_bytes
    if memory_limit_mb:
        _batch_optimizer.memory_limit_bytes = memory_limit_mb * 1024 * 1024

    # Start monitoring if not already running
    was_monitoring = _memory_monitor._monitoring
    if not was_monitoring:
        _memory_monitor.start_monitoring()

    try:
        yield
    finally:
        # Restore original memory limit
        _batch_optimizer.memory_limit_bytes = original_limit

        # Stop monitoring if we started it
        if not was_monitoring:
            _memory_monitor.stop_monitoring()

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_memory_pool() -> MemoryPool:
    """Get the global memory pool instance."""
    return _memory_pool


def get_batch_optimizer() -> BatchSizeOptimizer:
    """Get the global batch optimizer instance."""
    return _batch_optimizer


def get_memory_monitor() -> MemoryMonitor:
    """Get the global memory monitor instance."""
    return _memory_monitor


def get_cycle_detector() -> ReferenceCycleDetector:
    """Get the global reference cycle detector instance."""
    return _cycle_detector


def optimize_model_memory(model: nn.Module) -> None:
    """
    Apply memory optimizations to a PyTorch model.

    Args:
        model: PyTorch model to optimize
    """
    # Enable gradient checkpointing if available
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")

    # Set model to evaluation mode for memory efficiency
    model.eval()

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_stats() -> Dict[str, Any]:
    """Get comprehensive memory statistics."""
    stats = {
        "memory_pool": _memory_pool.get_stats(),
        "memory_monitor": {
            "current_usage": _memory_monitor.get_current_memory_usage(),
            "peak_usage": _memory_monitor.get_peak_memory_usage(),
            "alerts": _memory_monitor.get_memory_alerts(),
        },
        "cycle_detector": {
            "tracked_objects": _cycle_detector.get_tracked_count(),
            "cycles_detected": _cycle_detector._cycle_count,
        },
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        stats["cuda"] = {
            "total_memory": props.total_memory,
            "allocated_memory": torch.cuda.memory_allocated(),
            "cached_memory": torch.cuda.memory_reserved(),
            "utilization": (torch.cuda.memory_allocated() / props.total_memory) * 100,
        }

    return stats


def log_memory_summary() -> None:
    """Log a comprehensive memory summary."""
    stats = get_memory_stats()

    logger.info("=== Memory Summary ===")

    # Memory pool stats
    pool_stats = stats["memory_pool"]
    hit_rate = (
        pool_stats["hits"] / max(pool_stats["hits"] + pool_stats["misses"], 1) * 100
    )
    logger.info(
        f"Memory Pool: {hit_rate:.1f}% hit rate, "
        f"{pool_stats['pooled_tensors']} pooled tensors"
    )

    # Memory monitor stats
    monitor_stats = stats["memory_monitor"]
    current_mb = monitor_stats["current_usage"] / (1024 * 1024)
    peak_mb = monitor_stats["peak_usage"] / (1024 * 1024)
    logger.info(f"Memory Usage: {current_mb:.1f}MB current, {peak_mb:.1f}MB peak")

    if monitor_stats["alerts"]:
        logger.warning(f"Memory Alerts: {len(monitor_stats['alerts'])}")

    # CUDA stats if available
    if "cuda" in stats:
        cuda_stats = stats["cuda"]
        total_mb = cuda_stats["total_memory"] / (1024 * 1024)
        allocated_mb = cuda_stats["allocated_memory"] / (1024 * 1024)
        logger.info(
            f"CUDA: {allocated_mb:.1f}/{total_mb:.1f}MB "
            f"({cuda_stats['utilization']:.1f}% utilization)"
        )

    # Reference cycle stats
    cycle_stats = stats["cycle_detector"]
    if cycle_stats["cycles_detected"] > 0:
        logger.warning(
            f"Reference Cycles: {cycle_stats['cycles_detected']} objects detected"
        )

    logger.info("====================")
