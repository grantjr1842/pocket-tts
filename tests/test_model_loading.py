"""Tests for model loading functionality."""

import pytest
from pocket_tts import TTSModel


class TestModelLoading:
    """Test suite for TTS model loading functionality."""

    def test_load_model_basic(self):
        """Test basic model loading."""
        model = TTSModel.load_model()
        assert model is not None
        assert hasattr(model, "sample_rate")
        assert model.sample_rate > 0
        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_load_model_variant(self):
        """Test loading model with different variant."""
        model = TTSModel.load_model(variant="b6369a24")
        assert model is not None
        assert hasattr(model, "sample_rate")
        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_load_model_with_temp(self):
        """Test loading model with temperature parameter."""
        model = TTSModel.load_model(temp=0.8)
        assert model is not None
        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_load_model_with_lsd_steps(self):
        """Test loading model with LSD decode steps."""
        model = TTSModel.load_model(lsd_decode_steps=2)
        assert model is not None
        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_load_model_with_noise_clamp(self):
        """Test loading model with noise clamp."""
        model = TTSModel.load_model(noise_clamp=1.0)
        assert model is not None
        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_load_model_with_eos_threshold(self):
        """Test loading model with EOS threshold."""
        model = TTSModel.load_model(eos_threshold=-3.0)
        assert model is not None
        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_model_components_exist(self):
        """Test that all required model components exist."""
        model = TTSModel.load_model()

        # Check main components
        assert hasattr(model, "flow_lm")
        assert hasattr(model, "mimi_encoder")
        assert hasattr(model, "mimi_decoder")
        assert hasattr(model, "text_encoder")

        # Check that components are properly initialized
        assert model.flow_lm is not None
        assert model.mimi_encoder is not None
        assert model.mimi_decoder is not None
        assert model.text_encoder is not None

        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_model_sample_rate(self):
        """Test model sample rate configuration."""
        model = TTSModel.load_model()
        assert model.sample_rate == 24000  # Standard sample rate for Pocket TTS
        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_model_state_dict_structure(self):
        """Test that model state dict has expected structure."""
        model = TTSModel.load_model()
        state_dict = model.state_dict()

        # Check for expected keys in state dict
        expected_patterns = ["flow_lm", "mimi", "text_encoder"]
        for pattern in expected_patterns:
            assert any(pattern in key for key in state_dict.keys()), (
                f"Missing {pattern} in state dict"
            )

        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_model_inference_mode(self):
        """Test that model is in inference mode."""
        model = TTSModel.load_model()

        # Check that model is in evaluation mode
        assert not model.flow_lm.training
        assert not model.mimi_encoder.training
        assert not model.mimi_decoder.training
        assert not model.text_encoder.training

        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_model_caching_behavior(self):
        """Test model loading caching behavior."""
        # Clear any existing cache
        TTSModel._model_cache.clear()

        # Load model twice
        model1 = TTSModel.load_model()
        model2 = TTSModel.load_model()

        # Should be the same instance (cached)
        assert model1 is model2

        model1._cached_get_state_for_audio_prompt.cache_clear()

    def test_model_cache_clear(self):
        """Test model cache clearing functionality."""
        # Load model to populate cache
        model = TTSModel.load_model()
        assert len(TTSModel._model_cache) > 0

        # Clear cache
        TTSModel.clear_model_cache()
        assert len(TTSModel._model_cache) == 0

        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_load_model_with_invalid_device(self):
        """Test loading model with invalid device."""
        with pytest.raises(ValueError, match="Invalid device"):
            TTSModel.load_model(device="invalid_device")

    def test_model_parameters_count(self):
        """Test that model has expected number of parameters."""
        model = TTSModel.load_model()

        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 100_000_000  # Should be around 100M parameters

        model._cached_get_state_for_audio_prompt.cache_clear()

    def test_model_gradient_disabled(self):
        """Test that model has gradients disabled for inference."""
        model = TTSModel.load_model()

        # Check that gradients are disabled
        for param in model.parameters():
            assert not param.requires_grad

        model._cached_get_state_for_audio_prompt.cache_clear()
