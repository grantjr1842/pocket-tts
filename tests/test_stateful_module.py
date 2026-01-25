"""Tests for StatefulModule base class and its implementations."""

import pytest
import torch
import torch.nn as nn

from pocket_tts.modules.stateful_module import (
    StatefulModule,
    init_states,
    increment_steps,
    trim_model_state,
)


class MockStatefulModule(StatefulModule):
    """Mock implementation of StatefulModule for testing."""

    def __init__(self, state_size: int = 10):
        super().__init__()
        self.state_size = state_size
        self.linear = nn.Linear(5, 5)  # Add some parameters

    def init_state(
        self, batch_size: int, sequence_length: int
    ) -> dict[str, torch.Tensor]:
        """Initialize mock state with tensors."""
        return {
            "counter": torch.zeros(batch_size, dtype=torch.long),
            "cache": torch.zeros(batch_size, sequence_length, self.state_size),
            "metadata": torch.ones(batch_size, 3),
        }

    def increment_step(self, state: dict, increment: int = 1):
        """Increment counter by specified amount."""
        state["counter"] += increment
        # Mark some cache positions as used
        if "used_length" in state:
            state["used_length"] += increment
        else:
            state["used_length"] = torch.full_like(state["counter"], increment)


class TestStatefulModule:
    """Test cases for StatefulModule base class."""

    def test_abstract_methods(self):
        """Test that StatefulModule is properly abstract."""
        with pytest.raises(TypeError):
            StatefulModule()

    def test_concrete_implementation(self):
        """Test that concrete implementations work correctly."""
        module = MockStatefulModule(state_size=8)
        assert module.state_size == 8
        assert hasattr(module, "linear")

    def test_init_state_signature(self):
        """Test init_state method signature and return type."""
        module = MockStatefulModule()

        # Test with different batch sizes and sequence lengths
        test_cases = [
            (1, 10),
            (4, 20),
            (8, 5),
        ]

        for batch_size, seq_len in test_cases:
            state = module.init_state(batch_size, seq_len)

            # Check return type
            assert isinstance(state, dict)

            # Check tensor shapes
            assert "counter" in state
            assert "cache" in state
            assert "metadata" in state

            assert state["counter"].shape == (batch_size,)
            assert state["cache"].shape == (batch_size, seq_len, module.state_size)
            assert state["metadata"].shape == (batch_size, 3)

            # Check tensor types
            assert state["counter"].dtype == torch.long
            assert state["cache"].dtype == torch.float32
            assert state["metadata"].dtype == torch.float32

    def test_increment_step_default(self):
        """Test increment_step with default increment."""
        module = MockStatefulModule()
        state = module.init_state(2, 10)

        # Check initial state
        assert torch.all(state["counter"] == 0)

        # Increment by default (1)
        module.increment_step(state)
        assert torch.all(state["counter"] == 1)
        assert torch.all(state["used_length"] == 1)

        # Increment again
        module.increment_step(state)
        assert torch.all(state["counter"] == 2)
        assert torch.all(state["used_length"] == 2)

    def test_increment_step_custom(self):
        """Test increment_step with custom increment."""
        module = MockStatefulModule()
        state = module.init_state(3, 15)

        # Increment by custom amount
        increment = 5
        module.increment_step(state, increment)
        assert torch.all(state["counter"] == 5)
        assert torch.all(state["used_length"] == 5)

        # Increment again
        module.increment_step(state, 3)
        assert torch.all(state["counter"] == 8)
        assert torch.all(state["used_length"] == 8)

    def test_get_state(self):
        """Test get_state method."""
        module = MockStatefulModule()

        # Test without module name set
        model_state = {"test_module": {"test": torch.ones(5)}}
        with pytest.raises(RuntimeError, match="Module absolute name not set"):
            module.get_state(model_state)

        # Test with module name set
        module._module_absolute_name = "test_module"
        result = module.get_state(model_state)
        assert result == model_state["test_module"]
        assert "test" in result


class TestStatefulModuleFunctions:
    """Test cases for StatefulModule utility functions."""

    def test_init_states(self):
        """Test init_states function with multiple modules."""

        # Create a model with multiple StatefulModules
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.module1 = MockStatefulModule(state_size=5)
                self.module2 = MockStatefulModule(state_size=8)
                self.module3 = nn.Linear(10, 5)  # Not a StatefulModule

        model = TestModel()
        batch_size = 3
        seq_len = 12

        model_state = init_states(model, batch_size, seq_len)

        # Check that only StatefulModules are included
        assert len(model_state) == 2
        assert "module1" in model_state
        assert "module2" in model_state
        assert "module3" not in model_state

        # Check module names are set
        assert model.module1._module_absolute_name == "module1"
        assert model.module2._module_absolute_name == "module2"

        # Check state shapes
        assert model_state["module1"]["cache"].shape == (batch_size, seq_len, 5)
        assert model_state["module2"]["cache"].shape == (batch_size, seq_len, 8)

    def test_increment_steps(self):
        """Test increment_steps function."""

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.module1 = MockStatefulModule()
                self.module2 = MockStatefulModule()

        model = TestModel()
        model_state = init_states(model, 2, 10)

        # Check initial counters
        assert torch.all(model_state["module1"]["counter"] == 0)
        assert torch.all(model_state["module2"]["counter"] == 0)

        # Increment steps
        increment_steps(model, model_state, increment=3)

        # Check counters are incremented
        assert torch.all(model_state["module1"]["counter"] == 3)
        assert torch.all(model_state["module2"]["counter"] == 3)
        assert torch.all(model_state["module1"]["used_length"] == 3)
        assert torch.all(model_state["module2"]["used_length"] == 3)

    def test_trim_model_state(self):
        """Test trim_model_state function."""
        # Create mock state with cache
        batch_size = 2
        full_seq_len = 100
        used_len = 30
        hidden_dim = 8

        state = {
            "cache": torch.randn(
                2, batch_size, full_seq_len, hidden_dim
            ),  # [2, B, seq, H]
            "current_end": torch.arange(
                used_len, dtype=torch.long
            ),  # Used length marker
            "other": torch.ones(batch_size, 5),  # Non-cache tensor
        }

        model_state = {"test_module": state}

        # Trim the state
        trimmed_state = trim_model_state(model_state)

        # Check cache is trimmed
        assert trimmed_state["test_module"]["cache"].shape[2] == used_len
        assert trimmed_state["test_module"]["other"].shape == state["other"].shape

        # Check values are preserved for used portion
        assert torch.allclose(
            trimmed_state["test_module"]["cache"][:, :, :used_len, :],
            state["cache"][:, :, :used_len, :],
        )

    def test_trim_model_state_ring_buffer(self):
        """Test trim_model_state with ring buffer format."""
        batch_size = 2
        capacity = 100
        used_len = 25
        num_heads = 4
        hidden_dim = 8

        state = {
            "cache": torch.randn(
                2, batch_size, num_heads, capacity, hidden_dim
            ),  # Ring buffer format
            "end_offset": torch.full((batch_size,), used_len, dtype=torch.long),
            "other": torch.ones(batch_size, 3),
        }

        model_state = {"test_module": state}
        trimmed_state = trim_model_state(model_state)

        # Check cache is trimmed to used length
        assert trimmed_state["test_module"]["cache"].shape[3] == used_len
        assert trimmed_state["test_module"]["other"].shape == state["other"].shape


class TestStatefulModuleIntegration:
    """Integration tests for StatefulModule with real implementations."""

    def test_streaming_conv_integration(self):
        """Test integration with StreamingConv1d."""
        from pocket_tts.modules.conv import StreamingConv1d

        conv = StreamingConv1d(in_channels=3, out_channels=5, kernel_size=4, stride=2)

        # Test state initialization
        batch_size = 2
        seq_len = 10
        state = conv.init_state(batch_size, seq_len)

        assert "previous" in state
        assert "first" in state
        assert state["previous"].shape == (batch_size, 3, 2)  # kernel - stride = 4-2
        assert state["first"].shape == (batch_size,)
        assert torch.all(state["first"] == 1)

    def test_streaming_attention_integration(self):
        """Test integration with StreamingMultiheadAttention."""
        from pocket_tts.modules.streaming_attention import StreamingMultiheadAttention
        from pocket_tts.modules.rope import RotaryEmbedding

        embed_dim = 32
        num_heads = 4
        rope = RotaryEmbedding(embed_dim // num_heads)
        attn = StreamingMultiheadAttention(embed_dim, num_heads, rope)

        # Test state initialization
        batch_size = 3
        seq_len = 16
        state = attn.init_state(batch_size, seq_len)

        # Should have cache and current_end for append mode
        assert "cache" in state
        assert "current_end" in state
        assert state["cache"].shape == (
            2,
            batch_size,
            seq_len,
            num_heads,
            embed_dim // num_heads,
        )
        assert state["current_end"].shape == (0,)

        # Test increment step
        attn.increment_step(state, increment=5)
        assert state["current_end"].shape == (5,)

    def test_performance_large_batch(self):
        """Test performance with large batch sizes."""
        module = MockStatefulModule(state_size=64)

        # Test with large batch
        large_batch = 128
        seq_len = 512

        import time

        start_time = time.time()
        state = module.init_state(large_batch, seq_len)
        init_time = time.time() - start_time

        # Should complete quickly
        assert init_time < 1.0  # 1 second max

        # Test increment performance
        start_time = time.time()
        for _ in range(10):
            module.increment_step(state, increment=1)
        increment_time = time.time() - start_time

        assert increment_time < 0.1  # 100ms max for 10 increments

        # Check final state
        assert torch.all(state["counter"] == 10)


if __name__ == "__main__":
    pytest.main([__file__])
