"""Tests for numpy replacement functionality."""

import pytest
import numpy as np

# Test what's actually available
from pocket_tts import _RUST_NUMPY_AVAILABLE


class TestNumpyReplacement:
    """Test suite for numpy replacement functionality."""

    def test_rust_numpy_availability(self):
        """Test Rust numpy availability flag."""
        assert isinstance(_RUST_NUMPY_AVAILABLE, bool)

        if _RUST_NUMPY_AVAILABLE:
            # If Rust numpy is available, test the functions
            try:
                from pocket_tts import arange, array, min, max, mean, sum

                # Test basic functionality
                result = arange(0, 5)
                expected = np.arange(5)
                np.testing.assert_array_equal(result, expected)

                # Test array creation
                data = [1, 2, 3, 4, 5]
                result = array(data)
                expected = np.array(data)
                np.testing.assert_array_equal(result, expected)

                # Test statistical functions
                data = np.array([1, 2, 3, 4, 5])
                assert min(data) == np.min(data)
                assert max(data) == np.max(data)
                assert mean(data) == np.mean(data)
                assert sum(data) == np.sum(data)

            except ImportError:
                pytest.skip("Rust numpy functions not importable")
        else:
            # When Rust numpy is not available, functions should not be importable
            with pytest.raises(ImportError):
                from pocket_tts import arange

    def test_numpy_fallback_behavior(self):
        """Test that the system gracefully handles missing Rust numpy."""
        # This test verifies the system behavior when Rust numpy is not available

        if not _RUST_NUMPY_AVAILABLE:
            # The functions should not be available when Rust numpy is not available
            numpy_functions = [
                "arange",
                "array",
                "clip",
                "min",
                "max",
                "mean",
                "median",
                "sum",
                "sqrt",
                "log",
                "std",
                "var",
                "reshape",
                "transpose",
                "concatenate",
                "vstack",
                "hstack",
                "zeros",
                "ones",
                "eye",
                "linspace",
                "interp",
                "dot",
                "matmul",
                "abs",
                "power",
                "frombuffer",
                "size",
                "percentile",
            ]

            for func_name in numpy_functions:
                with pytest.raises(ImportError):
                    exec(f"from pocket_tts import {func_name}")
        else:
            # If Rust numpy is available, this test should pass
            pytest.skip("Rust numpy is available, fallback not needed")

    def test_core_functionality_available(self):
        """Test that core TTS functionality is available regardless of numpy implementation."""
        # These should always be available
        from pocket_tts import TTSModel, load_wav, save_audio

        assert TTSModel is not None
        assert load_wav is not None
        assert save_audio is not None

    def test_audio_functions_with_regular_numpy(self):
        """Test that audio functions work with regular numpy when Rust numpy is not available."""
        from pocket_tts import load_wav

        # Create a simple test using regular numpy
        import tempfile
        import wave

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            # Create a simple WAV file
            sample_rate = 24000
            samples = 1000
            audio_data = np.random.randint(-32768, 32767, samples, dtype=np.int16)

            with wave.open(temp_path, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())

            # Load the WAV file - this should work regardless of Rust numpy
            audio, sr = load_wav(temp_path)

            assert isinstance(audio, np.ndarray) or hasattr(
                audio, "numpy"
            )  # torch tensor or numpy array
            assert sr == sample_rate

        finally:
            import os

            if os.path.exists(temp_path):
                os.unlink(temp_path)
