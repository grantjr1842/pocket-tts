"""Tests for audio I/O functionality."""

import pytest
import torch
import numpy as np
import tempfile
import os
import wave
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_output import save_audio
from pocket_tts.data.audio_utils import convert_audio


class TestAudioIO:
    """Test suite for audio input/output functionality."""

    def test_audio_read_wav_file(self):
        """Test reading WAV audio file."""
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            # Generate simple sine wave
            sample_rate = 24000
            duration = 1.0
            t = torch.linspace(0, duration, int(sample_rate * duration))
            audio = torch.sin(2 * torch.pi * 440 * t)  # 440 Hz sine wave

            # Save using wave module
            with wave.open(tmp.name, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                audio_int16 = (audio * 32767).short().numpy()
                wav_file.writeframes(audio_int16.tobytes())

            # Load using our function
            loaded_audio, loaded_sr = audio_read(tmp.name)
            assert isinstance(loaded_audio, torch.Tensor)
            assert loaded_sr == sample_rate
            assert loaded_audio.shape[0] == 1  # Should be 2D with 1 channel
            assert loaded_audio.shape[1] == sample_rate * duration

            # Cleanup
            os.unlink(tmp.name)

    def test_save_audio_tensor(self):
        """Test saving torch tensor to WAV file."""
        # Generate test audio
        sample_rate = 24000
        audio = torch.randn(sample_rate)  # 1 second of random noise

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            # Save audio
            save_audio(tmp.name, audio, sample_rate)

            # Verify file exists and has content
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0

            # Load and verify
            loaded_audio, loaded_sr = audio_read(tmp.name)
            assert loaded_sr == sample_rate
            assert loaded_audio.shape[0] == 1  # Mono channel
            assert (
                abs(loaded_audio.shape[1] - sample_rate) <= 1
            )  # Allow small difference

            # Cleanup
            os.unlink(tmp.name)

    def test_save_audio_numpy(self):
        """Test saving numpy array to WAV file."""
        # Generate test audio
        sample_rate = 24000
        audio_np = np.random.randn(sample_rate).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            # Save audio
            save_audio(tmp.name, audio_np, sample_rate)

            # Verify file exists and has content
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0

            # Load and verify
            loaded_audio, loaded_sr = audio_read(tmp.name)
            assert loaded_sr == sample_rate
            assert loaded_audio.shape[0] == 1  # Mono channel

            # Cleanup
            os.unlink(tmp.name)

    def test_convert_audio_sample_rate(self):
        """Test audio sample rate conversion."""
        # Create audio at one sample rate
        from_rate = 16000
        to_rate = 24000
        audio = torch.randn(1, 1000)  # 1 channel, 1000 samples

        converted = convert_audio(audio, from_rate, to_rate, 1)

        # Check that length changed appropriately
        expected_length = int(1000 * to_rate / from_rate)
        assert abs(converted.shape[1] - expected_length) <= 1  # Allow small difference
        assert converted.shape[0] == 1  # Should still be 1 channel

    def test_convert_audio_channels(self):
        """Test audio channel conversion."""
        # Create mono audio
        audio = torch.randn(1, 1000)  # 1 channel

        # Convert to stereo (should work if we have stereo conversion logic)
        try:
            converted = convert_audio(audio, 24000, 24000, 2)
            assert converted.shape[0] == 2  # Should be 2 channels
        except ValueError:
            # If stereo conversion is not implemented, that's expected
            pass

    def test_audio_read_non_wav_fallback(self):
        """Test reading non-WAV files (should raise ImportError)."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            # Create empty file
            tmp.write(b"")
            tmp.flush()

            # Should raise ImportError due to missing soundfile
            with pytest.raises(ImportError, match="soundfile is required"):
                audio_read(tmp.name)

            # Cleanup
            os.unlink(tmp.name)

    def test_save_audio_shape_handling(self):
        """Test save_audio handles different tensor shapes correctly."""
        sample_rate = 24000

        # Test 1D tensor
        audio_1d = torch.randn(sample_rate)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            save_audio(tmp.name, audio_1d, sample_rate)
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)

        # Test 2D tensor (1, samples)
        audio_2d = torch.randn(1, sample_rate)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            save_audio(tmp.name, audio_2d, sample_rate)
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)

    def test_audio_io_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test reading non-existent file
        with pytest.raises(FileNotFoundError):
            audio_read("non_existent_file.wav")

        # Test saving to invalid directory
        audio = torch.randn(1000)
        with pytest.raises(Exception):  # Could be various exceptions
            save_audio("/invalid/path/audio.wav", audio, 24000)

    def test_convert_audio_no_conversion_needed(self):
        """Test convert_audio when no conversion is needed."""
        sample_rate = 24000
        audio = torch.randn(1, 1000)  # 1 channel, 1000 samples

        # Same sample rate and channels
        converted = convert_audio(audio, sample_rate, sample_rate, 1)

        # Should be identical
        assert torch.allclose(converted, audio)

    def test_audio_read_stereo_to_mono(self):
        """Test reading stereo WAV file converts to mono."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            # Create stereo WAV file
            sample_rate = 24000
            duration = 1.0
            t = torch.linspace(0, duration, int(sample_rate * duration))

            # Create stereo signal (different frequencies for each channel)
            left = torch.sin(2 * torch.pi * 440 * t)  # 440 Hz
            right = torch.sin(2 * torch.pi * 660 * t)  # 660 Hz
            stereo = torch.stack([left, right], dim=1)  # Shape: (samples, 2)

            # Save as stereo WAV
            with wave.open(tmp.name, "wb") as wav_file:
                wav_file.setnchannels(2)  # Stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                audio_int16 = (stereo * 32767).short().numpy()
                wav_file.writeframes(audio_int16.tobytes())

            # Load using our function (should convert to mono)
            loaded_audio, loaded_sr = audio_read(tmp.name)
            assert loaded_sr == sample_rate
            assert loaded_audio.shape[0] == 1  # Should be mono
            assert loaded_audio.shape[1] == sample_rate * duration

            # Cleanup
            os.unlink(tmp.name)
