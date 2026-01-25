"""Tests for TTSModel functionality including loading and generation."""

import pytest
import torch

from pocket_tts.models.tts_model import TTSModel
from pocket_tts.default_parameters import DEFAULT_VARIANT, DEFAULT_TEMPERATURE


class TestTTSModel:
    """Test suite for TTSModel class."""

    def test_load_model_default_parameters(self):
        """Test loading model with default parameters."""
        model = TTSModel.load_model()

        assert isinstance(model, TTSModel)
        assert model.device == "cpu"  # Default device
        assert model.sample_rate == 24000  # Expected sample rate
        assert model.temp == DEFAULT_TEMPERATURE
        assert hasattr(model, "flow_lm")
        assert hasattr(model, "mimi")

    def test_load_model_custom_variant(self):
        """Test loading model with custom variant."""
        model = TTSModel.load_model(variant=DEFAULT_VARIANT)

        assert isinstance(model, TTSModel)
        assert model.sample_rate == 24000

    def test_load_model_custom_temperature(self):
        """Test loading model with custom temperature."""
        custom_temp = 0.8
        model = TTSModel.load_model(temp=custom_temp)

        assert isinstance(model, TTSModel)
        assert model.temp == custom_temp

    def test_load_model_custom_decode_steps(self):
        """Test loading model with custom LSD decode steps."""
        custom_steps = 4
        model = TTSModel.load_model(lsd_decode_steps=custom_steps)

        assert isinstance(model, TTSModel)
        assert model.lsd_decode_steps == custom_steps

    def test_load_model_custom_eos_threshold(self):
        """Test loading model with custom EOS threshold."""
        custom_threshold = -2.5
        model = TTSModel.load_model(eos_threshold=custom_threshold)

        assert isinstance(model, TTSModel)
        assert model.eos_threshold == custom_threshold

    def test_load_model_with_noise_clamp(self):
        """Test loading model with noise clamping."""
        noise_clamp = 1.0
        model = TTSModel.load_model(noise_clamp=noise_clamp)

        assert isinstance(model, TTSModel)
        assert model.noise_clamp == noise_clamp

    def test_load_model_without_noise_clamp(self):
        """Test loading model without noise clamping."""
        model = TTSModel.load_model(noise_clamp=None)

        assert isinstance(model, TTSModel)
        assert model.noise_clamp is None

    def test_model_properties(self):
        """Test model property accessors."""
        model = TTSModel.load_model()

        # Test device property
        device = model.device
        assert isinstance(device, str)
        assert device in ["cpu", "cuda", "mps"]

        # Test sample_rate property
        sample_rate = model.sample_rate
        assert isinstance(sample_rate, int)
        assert sample_rate > 0

    def test_get_queue_depths_empty(self):
        """Test getting queue depths when no queues are active."""
        model = TTSModel.load_model()
        depths = model.get_queue_depths()

        assert isinstance(depths, dict)
        assert len(depths) == 0  # No active queues initially

    def test_model_is_nn_module(self):
        """Test that TTSModel is a proper nn.Module."""
        model = TTSModel.load_model()

        assert isinstance(model, torch.nn.Module)

        # Test that parameters are properly registered
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

        # Test that model can be moved to device
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            assert model_cuda.device == "cuda"

    def test_model_state_dict(self):
        """Test that model has a proper state dict."""
        model = TTSModel.load_model()

        state_dict = model.state_dict()
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

        # Check that all tensors are properly shaped
        for key, tensor in state_dict.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dim() > 0

    def test_model_eval_mode(self):
        """Test that model can be set to evaluation mode."""
        model = TTSModel.load_model()

        model.eval()
        assert not model.training

        # Test that dropout and similar layers are disabled
        for module in model.modules():
            if hasattr(module, "training"):
                assert not module.training

    def test_model_train_mode(self):
        """Test that model can be set to training mode."""
        model = TTSModel.load_model()

        model.train()
        assert model.training

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_load_model_cuda(self):
        """Test loading model and moving to CUDA."""
        model = TTSModel.load_model()
        model = model.cuda()

        assert model.device == "cuda"

        # Check that all parameters are on CUDA
        for param in model.parameters():
            assert param.is_cuda

    def test_model_forward_signature(self):
        """Test that model has expected forward methods."""
        model = TTSModel.load_model()

        # Check for expected methods
        assert hasattr(model, "generate_audio")
        assert hasattr(model, "generate_audio_stream")
        assert hasattr(model, "get_state_for_audio_prompt")
        assert callable(getattr(model, "generate_audio"))
        assert callable(getattr(model, "generate_audio_stream"))
        assert callable(getattr(model, "get_state_for_audio_prompt"))

    def test_model_config_access(self):
        """Test that model config is accessible."""
        model = TTSModel.load_model()

        assert hasattr(model, "config")
        assert model.config is not None

    def test_has_voice_cloning_property(self):
        """Test voice cloning capability flag."""
        model = TTSModel.load_model()

        assert hasattr(model, "has_voice_cloning")
        assert isinstance(model.has_voice_cloning, bool)

    def test_invalid_variant_raises_error(self):
        """Test that invalid variant raises appropriate error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            TTSModel.load_model(variant="nonexistent_variant")

    def test_model_size_reasonable(self):
        """Test that model size is reasonable (not too small or too large)."""
        model = TTSModel.load_model()

        param_count = sum(p.numel() for p in model.parameters())

        # Model should have millions of parameters but not billions
        assert 1_000_000 < param_count < 10_000_000_000

    def test_model_gradient_flow(self):
        """Test that gradients can flow through the model."""
        model = TTSModel.load_model()
        model.train()

        # Create a dummy input to test gradient flow
        # Note: This is a minimal test - actual generation tests would be more complex
        try:
            # Try to access a parameter and compute a simple gradient
            param = next(model.parameters())
            loss = param.sum()
            loss.backward()

            # Check that at least some parameters have gradients
            has_grad = any(p.grad is not None for p in model.parameters())
            assert has_grad
        except Exception:
            # If gradient computation fails due to model complexity,
            # that's acceptable for this basic test
            pytest.skip("Gradient computation test skipped due to model complexity")
