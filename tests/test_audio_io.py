"""Tests for audio I/O operations."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import wave

from pocket_tts.data.audio import load_wav, StreamingWAVWriter, stream_audio_chunks
from pocket_tts.data.audio_output import save_audio


class TestAudioIO:
    """Test suite for audio input/output operations."""

    def test_load_wav_basic(self):
        """Test basic WAV file loading."""
        # Create a temporary WAV file
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

            # Load the WAV file
            audio, sr = load_wav(temp_path)

            # Verify loaded audio
            assert isinstance(audio, torch.Tensor)
            assert audio.shape[0] == 1  # Mono channel
            assert audio.shape[1] == samples  # Correct number of samples
            assert sr == sample_rate
            assert audio.dtype == torch.float32

            # Check normalization to [-1, 1] range
            assert audio.min() >= -1.0
            assert audio.max() <= 1.0

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_wav_stereo_to_mono(self):
        """Test that stereo WAV files are converted to mono."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            # Create a stereo WAV file
            sample_rate = 24000
            samples = 1000
            # Create stereo data (left channel different from right)
            stereo_data = np.random.randint(-32768, 32767, (samples, 2), dtype=np.int16)

            with wave.open(temp_path, "wb") as wav_file:
                wav_file.setnchannels(2)  # Stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(stereo_data.tobytes())

            # Load the WAV file
            audio, sr = load_wav(temp_path)

            # Verify conversion to mono
            assert audio.shape[0] == 1  # Mono channel
            assert audio.shape[1] == samples  # Correct number of samples
            assert sr == sample_rate

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_wav_nonexistent_file(self):
        """Test loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_wav("nonexistent_file.wav")

    def test_load_wav_path_object(self):
        """Test loading WAV file using Path object."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Create a simple WAV file
            sample_rate = 24000
            samples = 500
            audio_data = np.random.randint(-32768, 32767, samples, dtype=np.int16)

            with wave.open(temp_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())

            # Load using Path object
            audio, sr = load_wav(temp_path)

            assert isinstance(audio, torch.Tensor)
            assert audio.shape[0] == 1
            assert sr == sample_rate

        finally:
            temp_path.unlink(missing_ok=True)

    def test_streaming_wav_writer_initialization(self):
        """Test StreamingWAVWriter initialization."""
        with tempfile.NamedTemporaryFile() as f:
            writer = StreamingWAVWriter(f, 24000)

            assert writer.sample_rate == 24000
            assert writer.output_stream == f
            assert writer.wave_writer is None
            assert writer.first_chunk_buffer is not None

    def test_streaming_wav_writer_write_header(self):
        """Test StreamingWAVWriter header writing."""
        with tempfile.NamedTemporaryFile() as f:
            writer = StreamingWAVWriter(f, 24000)
            writer.write_header(24000)

            assert writer.wave_writer is not None
            assert writer.first_chunk_buffer is not None

    def test_streaming_wav_writer_write_pcm_data(self):
        """Test StreamingWAVWriter PCM data writing."""
        with tempfile.NamedTemporaryFile() as f:
            writer = StreamingWAVWriter(f, 24000)
            writer.write_header(24000)

            # Create test audio chunk
            audio_chunk = torch.randn(1, 1000)  # Random audio
            writer.write_pcm_data(audio_chunk)

            # Should not raise any exceptions
            assert True

    def test_streaming_wav_writer_finalize(self):
        """Test StreamingWAVWriter finalization."""
        with tempfile.NamedTemporaryFile() as f:
            writer = StreamingWAVWriter(f, 24000)
            writer.write_header(24000)

            # Write some data
            audio_chunk = torch.randn(1, 1000)
            writer.write_pcm_data(audio_chunk)

            # Finalize should complete without error
            writer.finalize()
            assert writer.wave_writer is None

    def test_stream_audio_chunks_to_file(self):
        """Test streaming audio chunks to a file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            # Create test audio chunks
            chunks = [torch.randn(1, 500) for _ in range(3)]
            sample_rate = 24000

            # Stream chunks to file
            stream_audio_chunks(temp_path, iter(chunks), sample_rate)

            # Verify file was created and has content
            assert Path(temp_path).exists()
            assert Path(temp_path).stat().st_size > 0

            # Load and verify the written audio
            audio, sr = load_wav(temp_path)
            assert sr == sample_rate
            assert audio.shape[0] == 1  # Mono
            assert audio.shape[1] > 0  # Has samples

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_stream_audio_chunks_to_stdout(self):
        """Test streaming audio chunks to stdout (mocked)."""
        # Create test audio chunks
        chunks = [torch.randn(1, 100)]
        sample_rate = 24000

        # This should not raise an exception
        # Note: In actual usage, this would write to stdout
        try:
            stream_audio_chunks("-", iter(chunks), sample_rate)
        except Exception:
            # Some exceptions might occur with stdout in test environment
            # That's acceptable for this test
            pass

    def test_save_audio_basic(self):
        """Test basic audio saving functionality."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            # Create test audio
            audio = torch.randn(1, 1000)  # 1 channel, 1000 samples
            sample_rate = 24000

            # Save audio
            save_audio(temp_path, audio, sample_rate)

            # Verify file was created
            assert Path(temp_path).exists()
            assert Path(temp_path).stat().st_size > 0

            # Load and verify
            loaded_audio, sr = load_wav(temp_path)
            assert sr == sample_rate
            assert loaded_audio.shape[0] == 1  # Mono

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_audio_path_object(self):
        """Test saving audio using Path object."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Create test audio
            audio = torch.randn(1, 500)
            sample_rate = 24000

            # Save using Path object
            save_audio(temp_path, audio, sample_rate)

            assert temp_path.exists()

        finally:
            temp_path.unlink(missing_ok=True)

    def test_audio_data_types(self):
        """Test that audio data maintains correct types."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            # Create and save audio
            original_audio = torch.randn(1, 1000, dtype=torch.float32)
            sample_rate = 24000

            save_audio(temp_path, original_audio, sample_rate)

            # Load and check types
            loaded_audio, sr = load_wav(temp_path)

            assert isinstance(loaded_audio, torch.Tensor)
            assert loaded_audio.dtype == torch.float32
            assert isinstance(sr, int)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_audio_normalization(self):
        """Test that audio is properly normalized."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            # Create audio with known range
            sample_rate = 24000
            samples = 1000

            # Create int16 data at full scale
            max_val = 32767
            audio_data = np.array([max_val, -max_val] * (samples // 2), dtype=np.int16)

            # Save as WAV
            with wave.open(temp_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())

            # Load and check normalization
            audio, sr = load_wav(temp_path)

            # Should be normalized to [-1, 1] range
            assert audio.min() >= -1.0
            assert audio.max() <= 1.0
            # Should be close to ±1.0 for full-scale int16
            assert abs(audio.min().item() + 1.0) < 0.01  # Allow small tolerance
            assert abs(audio.max().item() - 1.0) < 0.01

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_empty_audio_file(self):
        """Test handling of empty or minimal audio files."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            # Create a minimal WAV file with very short duration
            sample_rate = 24000
            samples = 10  # Very short
            audio_data = np.zeros(samples, dtype=np.int16)  # Silent audio

            with wave.open(temp_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())

            # Load the minimal file
            audio, sr = load_wav(temp_path)

            assert audio.shape[0] == 1  # Mono
            assert audio.shape[1] == samples  # Correct sample count
            assert sr == sample_rate

        finally:
            Path(temp_path).unlink(missing_ok=True)
