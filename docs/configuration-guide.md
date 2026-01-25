# Pocket TTS Configuration Guide

This guide covers all configuration options for Pocket TTS, including environment variables, model parameters, compilation settings, voice configuration, and server deployment.

## Table of Contents

- [Environment Variables](#environment-variables)
- [Model Configuration](#model-configuration)
- [Compilation Settings](#compilation-settings)
- [Voice Configuration](#voice-configuration)
- [Server Configuration](#server-configuration)
- [Performance Tuning](#performance-tuning)
- [Platform-Specific Configuration](#platform-specific-configuration)

## Environment Variables

### Hugging Face Configuration

```bash
# Hugging Face token for accessing gated models
export HF_TOKEN="your_token_here"

# Hugging Face cache directory (default: ~/.cache/huggingface)
export HF_HOME="/path/to/cache"

# Hugging Face hub endpoint
export HF_ENDPOINT="https://huggingface.co"
```

**Example:**
```bash
# Set up Hugging Face authentication for voice cloning
export HF_TOKEN="hf_..."
uvx pocket-tts generate --voice "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
```

### PyTorch Configuration

```bash
# PyTorch thread count (default: all available cores)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Disable AVX/AVX2 if needed
export TORCH_DISABLE_AVX2=1

# Set PyTorch memory allocator behavior
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
```

**Example:**
```python
import os
# Limit PyTorch to use specific number of threads
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

from pocket_tts import TTSModel
model = TTSModel.load_model()
```

### Pocket TTS Specific Variables

```bash
# Default model variant
export POCKET_TTS_VARIANT="b6369a24"

# Default temperature
export POCKET_TTS_TEMP="0.7"

# Default decode steps
export POCKET_TTS_DECODE_STEPS="4"

# Default sample rate
export POCKET_TTS_SAMPLE_RATE="24000"

# Enable/disable verbose logging
export POCKET_TTS_VERBOSE="1"

# Cache directory for models and voices
export POCKET_TTS_CACHE_DIR="/path/to/cache"
```

**Example:**
```python
import os
# Configure via environment variables
os.environ['POCKET_TTS_TEMP'] = '0.5'
os.environ['POCKET_TTS_DECODE_STEPS'] = '8'
os.environ['POCKET_TTS_VERBOSE'] = '1'

from pocket_tts import TTSModel
model = TTSModel.load_model()  # Uses environment variables
```

### Server Configuration Variables

```bash
# Server host (default: localhost)
export POCKET_TTS_HOST="0.0.0.0"

# Server port (default: 8000)
export POCKET_TTS_PORT="8080"

# Enable auto-reload for development
export POCKET_TTS_RELOAD="1"

# CORS allowed origins
export POCKET_TTS_CORS_ORIGINS="http://localhost:3000,https://example.com"
```

**Example:**
```bash
# Start server with custom configuration
export POCKET_TTS_HOST="0.0.0.0"
export POCKET_TTS_PORT="9000"
uvx pocket-tts serve
```

## Model Configuration

### Model Variants

Pocket TTS supports different model variants with varying characteristics:

```python
from pocket_tts import TTSModel

# Default model (recommended)
model = TTSModel.load_model(variant="b6369a24")

# Alternative variants (if available)
# model = TTSModel.load_model(variant="alternative_variant")
```

### Generation Parameters

#### Temperature (`temp`)

Controls randomness in generation. Lower values produce more deterministic output.

```python
# Low temperature: more consistent, less expressive
model = TTSModel.load_model(temp=0.3)

# Medium temperature: balanced (default)
model = TTSModel.load_model(temp=0.7)

# High temperature: more expressive, varied
model = TTSModel.load_model(temp=1.0)

# Very high temperature: highly expressive, may be unstable
model = TTSModel.load_model(temp=1.5)
```

**Guidelines:**
- **0.3-0.5**: Professional, consistent output (audiobooks, announcements)
- **0.6-0.8**: Natural speech (general use)
- **0.9-1.2**: Expressive, emotional (storytelling, characters)
- **1.3+**: Highly experimental (may produce artifacts)

#### Decode Steps (`lsd_decode_steps`)

Number of generation iterations. More steps generally improve quality but increase generation time.

```python
# Fast generation (1 step)
model = TTSModel.load_model(lsd_decode_steps=1)

# Balanced speed/quality (4 steps, recommended)
model = TTSModel.load_model(lsd_decode_steps=4)

# High quality (8 steps)
model = TTSModel.load_model(lsd_decode_steps=8)

# Maximum quality (16 steps, slow)
model = TTSModel.load_model(lsd_decode_steps=16)
```

**Performance Impact:**
- 1 step: ~6x real-time (fastest)
- 4 steps: ~1.5x real-time (recommended)
- 8 steps: ~0.8x real-time (slower than real-time)
- 16 steps: ~0.4x real-time (slowest)

#### Noise Clamp (`noise_clamp`)

Limits extreme values in noise sampling to prevent audio artifacts.

```python
# No clamping (default)
model = TTSModel.load_model(noise_clamp=None)

# Conservative clamping (reduces artifacts)
model = TTSModel.load_model(noise_clamp=1.0)

# Aggressive clamping (may reduce expressiveness)
model = TTSModel.load_model(noise_clamp=0.5)
```

**Recommendations:**
- Use `None` for natural speech
- Use `1.0` if you hear clicks or pops
- Use `0.5-0.8` for very stable but less expressive output

#### EOS Threshold (`eos_threshold`)

Controls when the model detects end of speech. Lower values may produce longer audio.

```python
# Early termination (default)
model = TTSModel.load_model(eos_threshold=-4.0)

# Normal termination
model = TTSModel.load_model(eos_threshold=-5.0)

# Later termination (longer audio)
model = TTSModel.load_model(eos_threshold=-6.0)
```

## Compilation Settings

Pocket TTS supports `torch.compile()` for improved performance through model compilation.

### Basic Compilation

```python
from pocket_tts import TTSModel

model = TTSModel.load_model()
model.compile_for_inference()
```

### Compilation Options

#### Backend Selection

```python
# Inductor backend (default, recommended)
model.compile_for_inference(backend="inductor")

# AOT Autograd (experimental)
model.compile_for_inference(backend="aot_eager")

# Other backends (platform-dependent)
model.compile_for_inference(backend="cudagraphs")  # CUDA only
```

#### Compilation Mode

```python
# Reduce overhead (default, fastest)
model.compile_for_inference(mode="reduce-overhead")

# Default mode
model.compile_for_inference(mode="default")

# Max autotune (slower compilation, better runtime)
model.compile_for_inference(mode="max-autotune")
```

#### Advanced Compilation Options

```python
# Full graph capture (aggressive optimization)
model.compile_for_inference(
    fullgraph=True,
    mode="reduce-overhead"
)

# Dynamic shapes (flexible input sizes)
model.compile_for_inference(
    dynamic=True,
    mode="reduce-overhead"
)

# Selective compilation (compile specific components)
model.compile_for_inference(
    targets="flow-lm",  # Options: "all", "flow-lm", "mimi-decoder"
)

# Complete configuration
model.compile_for_inference(
    backend="inductor",
    mode="max-autotune",
    fullgraph=False,
    dynamic=False,
    targets="all"
)
```

### Compilation Trade-offs

| Configuration | Compilation Time | Runtime Speed | Memory Usage | Flexibility |
|---------------|------------------|---------------|--------------|-------------|
| No compilation | None | Baseline | Baseline | Full |
| Basic compile | ~10s | 1.3x faster | +50MB | Full |
| Max autotune | ~60s | 1.5x faster | +100MB | Full |
| Fullgraph | ~30s | 1.4x faster | +80MB | Limited |
| Dynamic | ~15s | 1.2x faster | +70MB | Full |

### CLI Compilation

```bash
# Enable compilation with defaults
pocket-tts generate --compile

# Custom compilation settings
pocket-tts generate \
  --compile \
  --compile-backend "inductor" \
  --compile-mode "max-autotune" \
  --compile-fullgraph

# Server with compilation
pocket-tts serve --compile --compile-mode "reduce-overhead"
```

## Voice Configuration

### Predefined Voices

Pocket TTS includes several predefined voices:

```python
from pocket_tts import TTSModel

model = TTSModel.load_model()

# Available voices
voices = {
    "alba": "hf://kyutai/tts-voices/alba-mackenna/casual.wav",
    "marius": "hf://kyutai/tts-voices/voice-donations/Selfie.wav",
    "javert": "hf://kyutai/tts-voices/voice-donations/Butter.wav",
    "jean": "hf://kyutai/tts-voices/ears/p010/freeform_speech_01.wav",
    "fantine": "hf://kyutai/tts-voices/vctk/p244_023.wav",
    "cosette": "hf://kyutai/tts-voices/expresso/ex04-ex02_confused_001_channel1_499s.wav",
    "eponine": "hf://kyutai/tts-voices/vctk/p262_023.wav",
    "azelma": "hf://kyutai/tts-voices/vctk/p303_023.wav",
}

# Load voice state
voice_state = model.get_state_for_audio_prompt(voices["alba"])
```

### Custom Voice Files

```python
# From local file
voice_state = model.get_state_for_audio_prompt("./my_voice.wav")

# From HTTP URL
voice_state = model.get_state_for_audio_prompt(
    "https://example.com/voice.wav"
)

# From Hugging Face
voice_state = model.get_state_for_audio_prompt(
    "hf://username/repo/voice.wav"
)
```

### Voice State Truncation

For long audio prompts, use truncation to limit memory usage:

```python
# No truncation (default)
voice_state = model.get_state_for_audio_prompt(
    "long_audio.wav",
    truncate=False
)

# Truncate to ~30 seconds (recommended for long audio)
voice_state = model.get_state_for_audio_prompt(
    "long_audio.wav",
    truncate=True
)
```

### Voice Quality Guidelines

For best results, custom voice files should:

- **Duration**: 2-30 seconds (optimal: 5-15 seconds)
- **Sample Rate**: 16kHz or higher (will be resampled to 24kHz)
- **Format**: WAV, MP3, or other audio formats supported by `torchaudio`
- **Content**: Clear speech with minimal background noise
- **Volume**: Moderate RMS level (0.2-0.7)

```python
from pocket_tts import load_wav, compute_audio_metrics

def check_voice_quality(audio_path):
    """Check if voice audio meets quality guidelines."""
    audio, sr = load_wav(audio_path)
    duration = audio.shape[1] / sr
    metrics = compute_audio_metrics(audio.numpy())

    print(f"Duration: {duration:.1f}s (recommended: 2-30s)")
    print(f"Sample Rate: {sr} Hz (recommended: 16000+)")
    print(f"RMS Level: {metrics['rms']:.2f} (recommended: 0.2-0.7)")
    print(f"Dynamic Range: {metrics['dynamic_range_db']:.1f} dB")

    if duration < 2:
        print("⚠️  Audio too short - may not capture voice characteristics")
    if duration > 30:
        print("⚠️  Audio very long - consider truncation")
    if metrics['rms'] < 0.1:
        print("⚠️  Audio very quiet - may affect cloning quality")
    if metrics['rms'] > 0.9:
        print("⚠️  Audio may be clipped")

check_voice_quality("my_voice.wav")
```

### Voice Caching

Cache frequently used voice states for better performance:

```python
from functools import lru_cache

class VoiceCache:
    def __init__(self, model):
        self.model = model
        self._cache = {}

    def get_voice(self, voice_name_or_path):
        """Get cached voice state."""
        if voice_name_or_path not in self._cache:
            self._cache[voice_name_or_path] = self.model.get_state_for_audio_prompt(
                voice_name_or_path
            )
        return self._cache[voice_name_or_path]

# Use voice cache
model = TTSModel.load_model()
voice_cache = VoiceCache(model)

alba_voice = voice_cache.get_voice("alba")
custom_voice = voice_cache.get_voice("./my_voice.wav")
```

## Server Configuration

### Basic Server Setup

```python
from pocket_tts import serve
import uvicorn

# Start server programmatically
uvicorn.run(
    serve.app,
    host="localhost",
    port=8000
)
```

### Advanced Server Configuration

```python
import uvicorn

# Custom server configuration
uvicorn.run(
    serve.app,
    host="0.0.0.0",  # Listen on all interfaces
    port=9000,
    log_level="info",
    access_log=True,
    reload=False,  # Disable auto-reload in production
    workers=1,  # Single worker (model is not thread-safe)
    limit_concurrency=5,  # Limit concurrent requests
    timeout_keep_alive=30,
)
```

### Production Server with Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn pocket_tts.serve:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install pocket-tts
RUN pip install --no-cache-dir pocket-tts

# Expose port
EXPOSE 8000

# Set environment variables
ENV POCKET_TTS_HOST=0.0.0.0
ENV POCKET_TTS_PORT=8000
ENV POCKET_TTS_RELOAD=0

# Run server
CMD ["pocket-tts", "serve"]
```

Build and run:

```bash
# Build image
docker build -t pocket-tts-server .

# Run container
docker run -p 8000:8000 pocket-tts-server

# Run with custom voice
docker run -p 8000:8000 \
  -v /path/to/voices:/voices \
  -e POCKET_TTS_VOICE=/voices/custom.wav \
  pocket-tts-server
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  pocket-tts:
    build: .
    ports:
      - "8000:8000"
    environment:
      - POCKET_TTS_HOST=0.0.0.0
      - POCKET_TTS_PORT=8000
      - POCKET_TTS_RELOAD=0
    volumes:
      - ./voices:/voices:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run:

```bash
docker-compose up -d
```

### Load Balancing

For high-traffic deployments, use a load balancer:

```yaml
# docker-compose.yml with multiple instances
version: '3.8'

services:
  pocket-tts-1:
    build: .
    ports:
      - "8001:8000"
    environment:
      - POCKET_TTS_PORT=8000

  pocket-tts-2:
    build: .
    ports:
      - "8002:8000"
    environment:
      - POCKET_TTS_PORT=8000

  nginx:
    image: nginx:alpine
    ports:
      - "8000:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - pocket-tts-1
      - pocket-tts-2
```

Nginx configuration (`nginx.conf`):

```nginx
events {
    worker_connections 1024;
}

http {
    upstream pocket_tts_backend {
        least_conn;
        server pocket-tts-1:8000;
        server pocket-tts-2:8000;
    }

    server {
        listen 80;

        location / {
            proxy_pass http://pocket_tts_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 300s;
        }
    }
}
```

## Performance Tuning

### CPU Optimization

```python
import os
import torch

# Limit CPU threads (recommended: 2-4)
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
torch.set_num_threads(4)

# Load model
from pocket_tts import TTSModel
model = TTSModel.load_model()
```

### Memory Optimization

```python
# Use streaming for long texts to reduce memory
for chunk in model.generate_audio_stream(voice_state, long_text):
    # Process chunk immediately
    pass

# Truncate voice states for memory efficiency
voice_state = model.get_state_for_audio_prompt(long_audio, truncate=True)

# Clear unused objects
import gc
del audio
gc.collect()
```

### Generation Speed Optimization

```python
# Fastest settings (lower quality)
model = TTSModel.load_model(
    temp=0.7,
    lsd_decode_steps=1,
    noise_clamp=1.0
)

# Balanced settings (recommended)
model = TTSModel.load_model(
    temp=0.7,
    lsd_decode_steps=4,
    noise_clamp=None
)

# High quality settings (slower)
model = TTSModel.load_model(
    temp=0.8,
    lsd_decode_steps=8,
    noise_clamp=None
)

# Compile for faster inference
model.compile_for_inference(mode="reduce-overhead")
```

## Platform-Specific Configuration

### Linux

```bash
# Install audio libraries
sudo apt-get install libsndfile1 libsndfile1-dev

# Optimize for CPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Run with proper thread affinity
taskset -c 0-3 pocket-tts serve  # Use specific CPU cores
```

### macOS

```bash
# Use conda for better compatibility
conda install pytorch torchvision torchaudio -c pytorch
pip install pocket-tts

# Optimize for Apple Silicon
export OMP_NUM_THREADS=8
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Windows

```bash
# Install Visual C++ Redistributable if needed
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

# Use CPU-only PyTorch for stability
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pocket-tts

# Set thread limits
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
```

## Configuration Examples

### Development Configuration

```python
# config/dev.py
import os

os.environ['POCKET_TTS_VERBOSE'] = '1'
os.environ['POCKET_TTS_TEMP'] = '0.7'
os.environ['POCKET_TTS_DECODE_STEPS'] = '1'  # Fast for iteration
```

### Production Configuration

```python
# config/prod.py
import os

os.environ['POCKET_TTS_VERBOSE'] = '0'
os.environ['POCKET_TTS_TEMP'] = '0.7'
os.environ['POCKET_TTS_DECODE_STEPS'] = '4'  # Balanced quality
os.environ['OMP_NUM_THREADS'] = '4'
```

### High-Quality Configuration

```python
# config/high_quality.py
import os

os.environ['POCKET_TTS_TEMP'] = '0.8'
os.environ['POCKET_TTS_DECODE_STEPS'] = '8'
os.environ['POCKET_TTS_CACHE_DIR'] = '/path/to/fast/cache'
```

## Debugging Configuration

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check configuration
from pocket_tts import TTSModel
import torch

model = TTSModel.load_model()
print(f"Device: {model.device}")
print(f"Sample Rate: {model.sample_rate}")
print(f"PyTorch Threads: {torch.get_num_threads()}")
print(f"PyTorch Version: {torch.__version__}")
```

For more information, see:
- [Python API Documentation](python-api.md)
- [Performance Optimization Guide](performance-optimization-guide.md)
- [Troubleshooting Guide](troubleshooting.md)
