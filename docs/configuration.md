# Configuration Guide

This guide covers all configuration options available for Pocket TTS, including environment variables, model parameters, and runtime settings.

## Environment Variables

### Model Configuration

```bash
# Cache size for voice prompts (default: 100)
export POCKET_TTS_PROMPT_CACHE_SIZE=50

# Custom model cache directory
export POCKET_TTS_CACHE_DIR="/path/to/cache"

# Disable Rust acceleration (fallback to Python)
export POCKET_TTS_DISABLE_RUST=1

# Audio processing buffer size
export POCKET_TTS_BUFFER_SIZE=8192
```

### Performance Configuration

```bash
# Number of CPU threads for PyTorch (default: 1)
export POCKET_TTS_THREADS=2

# Memory optimization level (0-3, default: 1)
export POCKET_TTS_MEMORY_LEVEL=2
```

## Model Parameters

### Core Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|--------|-------------|
| `variant` | str | "b6369a24" | Model signature/variant |
| `temperature` | float | 0.7 | 0.1-2.0 | Sampling temperature for generation |
| `lsd_decode_steps` | int | 1 | 1-10 | Number of generation steps |
| `noise_clamp` | float | None | 0.0-10.0 | Maximum noise value |
| `eos_threshold` | float | -4.0 | -10.0 to 0.0 | EOS detection threshold |
| `frames_after_eos` | int | None | 0-20 | Frames after EOS detection |

### Device Configuration

```python
from pocket_tts import TTSModel

# CPU (default, recommended)
model = TTSModel.load_model(device="cpu")

# GPU (experimental, may not provide speedup)
model = TTSModel.load_model(device="cuda")

# Auto-detect
model = TTSModel.load_model(device="auto")
```

## Compilation Options

PyTorch compilation can significantly improve performance:

```python
from pocket_tts import TTSModel

model = TTSModel.load_model()

# Basic compilation
model.compile_for_inference()

# Advanced compilation settings
model.compile_for_inference(
    backend="inductor",           # Compilation backend
    mode="reduce-overhead",       # Optimization mode
    fullgraph=True,              # Full graph capture
    dynamic=False,               # Static shapes
    targets=["flow-lm", "mimi-decoder"]  # Specific modules
)
```

### Compilation Backends

| Backend | Description | Use Case |
|---------|-------------|----------|
| `inductor` | Default, balanced performance | General use |
| `aot_eager` | Ahead-of-time compilation | Production |
| `cudagraphs` | CUDA-specific optimization | GPU only |

## Voice Configuration

### Built-in Voices

```python
# Available preset voices
voices = {
    "alba": "hf://kyutai/tts-voices/alba-mackenna/casual.wav",
    "marius": "hf://kyutai/tts-voices/voice-donations/Selfie.wav",
    "javert": "hf://kyutai/tts-voices/voice-donations/Butter.wav",
    "jean": "hf://kyutai/tts-voices/ears/p010/freeform_speech_01.wav",
    "fantine": "hf://kyutai/tts-voices/vctk/p244_023.wav",
    "cosette": "hf://kyutai/tts-voices/expresso/ex04-ex02_confused_001_channel1_499s.wav",
    "eponine": "hf://kyutai/tts-voices/vctk/p262_023.wav",
    "azelma": "hf://kyutai/tts-voices/vctk/p303_023.wav"
}
```

### Custom Voice Configuration

```python
from pocket_tts import TTSModel

model = TTSModel.load_model()

# Local file
voice_state = model.get_state_for_audio_prompt("./custom_voice.wav")

# Remote URL
voice_state = model.get_state_for_audio_prompt(
    "https://example.com/voice.wav"
)

# HuggingFace repository
voice_state = model.get_state_for_audio_prompt(
    "hf://username/repo/path/to/voice.wav"
)

# With truncation for long audio
voice_state = model.get_state_for_audio_prompt(
    "./long_voice.wav", 
    truncate=True
)
```

## Server Configuration

### Basic Server Options

```bash
pocket-tts serve \
    --host "0.0.0.0" \
    --port 8080 \
    --voice "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
```

### Advanced Server Configuration

```bash
# With compilation and custom settings
pocket-tts serve \
    --host "localhost" \
    --port 8000 \
    --compile \
    --compile-backend "inductor" \
    --compile-mode "reduce-overhead" \
    --compile-fullgraph \
    --voice "./custom_voice.wav"
```

## Audio Output Configuration

### Sample Rates and Formats

```python
from pocket_tts import save_audio, TTSModel
import torch

model = TTSModel.load_model()
voice_state = model.get_state_for_audio_prompt("alba")
audio = model.generate_audio(voice_state, "Hello world")

# Save with default settings (24kHz, 16-bit PCM)
save_audio("output.wav", audio, model.sample_rate)

# Save with custom sample rate
save_audio("output_48khz.wav", audio, 48000)
```

### Audio Processing Options

```python
from pocket_tts import (
    normalize_audio, 
    apply_gain, 
    resample_audio, 
    apply_fade
)

# Normalize audio to -1.0 to 1.0
normalized = normalize_audio(audio)

# Apply gain in dB
amplified = apply_gain(audio, 6.0)  # +6dB

# Resample to different rate
resampled = resample_audio(audio, 24000, 44100)

# Apply fade in/out
faded = apply_fade(audio, fade_in_samples=1000, fade_out_samples=1000)
```

## Troubleshooting Configuration

### Common Issues

**Issue**: Model loading fails
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
export POCKET_TTS_CACHE_DIR="./temp_cache"
```

**Issue**: Poor audio quality
```python
# Adjust generation parameters
model = TTSModel.load_model(
    temperature=0.5,           # Lower for more stable output
    lsd_decode_steps=5,         # More steps for quality
    eos_threshold=-3.0          # Earlier stopping
)
```

**Issue**: Slow performance
```python
# Enable compilation and optimize threads
model.compile_for_inference(mode="reduce-overhead")
import torch
torch.set_num_threads(2)  # Optimize for your CPU
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
model = TTSModel.load_model()
# Check device and settings
print(f"Device: {model.device}")
print(f"Sample rate: {model.sample_rate}")
print(f"Rust acceleration: {_RUST_NUMPY_AVAILABLE}")
```