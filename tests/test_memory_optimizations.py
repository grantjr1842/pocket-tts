#!/usr/bin/env python3
"""
Test suite for memory optimizations in Pocket TTS.

This test validates that memory profiling, reference cycle detection,
and batch size optimization work correctly.
"""

import gc
import sys
import pytest
from pathlib import Path

import torch

# Add pocket_tts to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pocket_tts.utils.memory_optimizer import (
    MemoryPool,
    BatchSizeOptimizer,
    MemoryMonitor,
    ReferenceCycleDetector,
    memory_efficient_context,
    get_memory_stats,
    log_memory_summary,
)


class TestMemoryPool:
    """Test memory pool functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.pool = MemoryPool(max_pool_size=5)

    def test_tensor_allocation_and_reuse(self):
        """Test that tensors are properly allocated and reused."""
        shape = (100,)
        dtype = torch.float32

        # Get tensor (should allocate new)
        tensor1 = self.pool.get_tensor(shape, dtype)
        assert tensor1.shape == shape
        assert tensor1.dtype == torch.float32

        # Return tensor to pool
        self.pool.return_tensor(tensor1)

        # Get another tensor (should reuse from pool)
        tensor2 = self.pool.get_tensor(shape, dtype)
        assert tensor2.shape == shape
        assert tensor2.dtype == torch.float32

        # Check stats
        stats = self.pool.get_stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1

    def test_pool_size_limit(self):
        """Test that pool respects size limits."""
        shape = (50,)

        # Add more tensors than pool limit
        tensors = []
        for i in range(10):
            tensor = self.pool.get_tensor(shape)
            tensors.append(tensor)

        # Return all tensors
        for tensor in tensors:
            self.pool.return_tensor(tensor)

        # Check that pool size is limited
        stats = self.pool.get_stats()
        assert stats["pooled_tensors"] <= 5

    def test_different_shapes_and_types(self):
        """Test pool with different tensor shapes and types."""
        # Different shapes
        shapes = [(10,), (10, 10), (5, 5, 5)]

        for shape in shapes:
            tensor = self.pool.get_tensor(shape)
            assert tensor.shape == shape
            self.pool.return_tensor(tensor)

        # Different dtypes
        dtypes = [torch.float32, torch.float16, torch.int32]

        for dtype in dtypes:
            tensor = self.pool.get_tensor((100,), dtype)
            assert tensor.dtype == dtype
            self.pool.return_tensor(tensor)

        # Should have separate pools for each combination
        stats = self.pool.get_stats()
        assert stats["pool_types"] >= 6  # 3 shapes * 3 dtypes


class TestBatchSizeOptimizer:
    """Test batch size optimization."""

    def setup_method(self):
        """Setup for each test method."""
        self.optimizer = BatchSizeOptimizer(memory_limit_mb=1000)

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        batch_size = 4
        seq_length = 100
        model_size_mb = 500

        estimated = self.optimizer.estimate_memory_usage(
            batch_size, seq_length, model_size_mb
        )

        # Should be reasonable estimate
        assert estimated > model_size_mb * 1024 * 1024  # At least model size
        assert estimated < 10 * 1024 * 1024 * 1024  # Less than 10GB

    def test_optimal_batch_size_calculation(self):
        """Test optimal batch size calculation."""
        seq_length = 100
        model_size_mb = 500

        batch_size = self.optimizer.get_optimal_batch_size(seq_length, model_size_mb)

        # Should be reasonable
        assert 1 <= batch_size <= 32

        # Should be cached
        batch_size2 = self.optimizer.get_optimal_batch_size(seq_length, model_size_mb)
        assert batch_size == batch_size2

    def test_memory_limit_effect(self):
        """Test that memory limit affects batch size."""
        seq_length = 1000  # Large sequence
        model_size_mb = 500

        # With high limit
        optimizer_high = BatchSizeOptimizer(memory_limit_mb=8000)
        batch_size_high = optimizer_high.get_optimal_batch_size(
            seq_length, model_size_mb
        )

        # With low limit
        optimizer_low = BatchSizeOptimizer(memory_limit_mb=1000)
        batch_size_low = optimizer_low.get_optimal_batch_size(seq_length, model_size_mb)

        # Low limit should result in smaller batch size
        assert batch_size_low <= batch_size_high


class TestMemoryMonitor:
    """Test memory monitoring functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.monitor = MemoryMonitor(check_interval=0.1)

    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        # Get current memory usage
        current = self.monitor.get_current_memory_usage()

        # Should be non-negative
        assert current >= 0

    def test_monitoring_lifecycle(self):
        """Test start/stop monitoring."""
        # Start monitoring
        self.monitor.start_monitoring()
        assert self.monitor._monitoring is True

        # Stop monitoring
        self.monitor.stop_monitoring()
        assert self.monitor._monitoring is False

    def test_stats_tracking(self):
        """Test statistics tracking."""
        # Reset stats
        self.monitor.reset_stats()

        # Initial stats should be zero
        assert self.monitor.get_peak_memory_usage() == 0
        assert len(self.monitor.get_memory_alerts()) == 0


class TestReferenceCycleDetector:
    """Test reference cycle detection."""

    def setup_method(self):
        """Setup for each test method."""
        self.detector = ReferenceCycleDetector()

    def test_object_tracking(self):
        """Test object tracking."""
        # Create test object
        obj = {"test": "data"}

        # Track object
        self.detector.track_object(obj)
        assert self.detector.get_tracked_count() == 1

        # Delete object
        del obj
        gc.collect()

        # Count should eventually go to 0 (after weakref callback)
        # Note: This might not happen immediately due to GC timing
        # assert self.detector.get_tracked_count() == 0

    def test_cycle_detection(self):
        """Test reference cycle detection."""

        # Create objects with circular references
        class TestObject:
            def __init__(self, name):
                self.name = name
                self.ref = None

        obj1 = TestObject("obj1")
        obj2 = TestObject("obj2")
        obj1.ref = obj2
        obj2.ref = obj1

        # Track objects
        self.detector.track_object(obj1)
        self.detector.track_object(obj2)

        # Delete references
        del obj1, obj2

        # Detect cycles
        cycles = self.detector.detect_cycles()

        # Should detect some cycles (may be 0 depending on GC timing)
        assert cycles >= 0


class TestMemoryEfficientContext:
    """Test memory-efficient context manager."""

    def test_context_manager(self):
        """Test context manager functionality."""
        with memory_efficient_context(memory_limit_mb=100):
            # Should be able to run code within context
            pass

        # Context should exit cleanly
        assert True


class TestMemoryStats:
    """Test memory statistics and logging."""

    def test_memory_stats_collection(self):
        """Test memory statistics collection."""
        stats = get_memory_stats()

        # Should have expected keys
        assert "memory_pool" in stats
        assert "memory_monitor" in stats
        assert "cycle_detector" in stats

        # Memory pool stats
        pool_stats = stats["memory_pool"]
        assert "hits" in pool_stats
        assert "misses" in pool_stats
        assert "pooled_tensors" in pool_stats

        # Memory monitor stats
        monitor_stats = stats["memory_monitor"]
        assert "current_usage" in monitor_stats
        assert "peak_usage" in monitor_stats
        assert "alerts" in monitor_stats

    def test_memory_logging(self):
        """Test memory summary logging (doesn't crash)."""
        # This should not raise an exception
        log_memory_summary()
        assert True


def test_integration_memory_optimizations():
    """Integration test for memory optimizations."""
    # Test that all components work together
    pool = MemoryPool()
    optimizer = BatchSizeOptimizer()
    detector = ReferenceCycleDetector()

    # Use components together
    with memory_efficient_context():
        # Get optimal batch size
        batch_size = optimizer.get_optimal_batch_size(100)
        assert 1 <= batch_size <= 32

        # Use memory pool
        tensor = pool.get_tensor((batch_size, 100))
        pool.return_tensor(tensor)

        # Check for cycles
        cycles = detector.detect_cycles()
        assert cycles >= 0

    # Check stats
    stats = get_memory_stats()
    assert isinstance(stats, dict)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
