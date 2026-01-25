# Pocket TTS - Comprehensive Technical Guide

*Generated for Kyutai Labs Pocket TTS Repository*
*Repository: https://github.com/kyutai-labs/pocket-tts*

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Model Architecture](#model-architecture)
3. [Python API Usage](#python-api-usage)
4. [Voice Cloning Implementation](#voice-cloning-implementation)
5. [Setup and Installation](#setup-and-installation)
6. [CLI Usage](#cli-usage)
7. [Advanced Configuration](#advanced-configuration)
8. [Development Guide](#development-guide)
9. [Performance Characteristics](#performance-characteristics)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

**Pocket TTS** is a lightweight, CPU-optimized text-to-speech model developed by Kyutai Labs. It represents a breakthrough in efficient TTS systems, running entirely on CPU hardware while maintaining high-quality output with voice cloning capabilities.

### Key Features

- **CPU-Optimized**: Runs efficiently on standard CPUs without requiring GPU acceleration
- **Small Model Size**: Only 100M parameters, making it lightweight and fast
- **Voice Cloning**: Clone voices from short audio clips (3-10 seconds)
- **Audio Streaming**: Low latency (~200ms to first audio chunk)
- **Fast Generation**: ~6x faster than real-time on MacBook Air M4
- **Streaming Architecture**: Generates audio frame-by-frame for real-time applications
- **English Support**: Currently supports English language synthesis
- **Infinite Text Handling**: Can process arbitrarily long text inputs

### Technical Specifications

| Specification | Value |
|--------------|-------|
| Parameters | 100M |
| Sample Rate | 24,000 Hz |
| Frame Rate | 12.5 Hz (80ms per frame) |
| Supported Python Versions | 3.10 - 3.14 |
| Required PyTorch Version | 2.5+ |
| Default Device | CPU |
| Bit Depth | 16-bit PCM |
| Output Format | WAV (Mono) |

### Resources

- **Live Demo**: https://kyutai.org/tts
- **Hugging Face Model**: https://huggingface.co/kyutai/pocket-tts
- **Research Paper**: https://arxiv.org/abs/2509.06926
- **Technical Report**: https://kyutai.org/pocket-tts-technical-report

---

## Model Architecture

The Pocket TTS architecture is built around four main components working in a streaming pipeline:

`★ Insight ─────────────────────────────────────`
**Architecture Innovation**:
- **Flow-Based Generation**: Uses Lagrangian Self Distillation (LSD) instead of traditional diffusion or autoregressive methods, enabling faster sampling with fewer steps
- **Neural Audio Codec**: Mimi codec compresses audio 48x (24kHz → 12.5Hz latent space), making computation extremely efficient
- **Streaming Design**: All modules are stateful, enabling frame-by-frame generation without processing the entire sequence at once
- **Dual-Thread Pipeline**: Generation and decoding run in parallel, with one thread generating latents while another decodes them to audio
`─────────────────────────────────────────────────`

### High-Level Architecture Diagram

```
┌─────────────────┐
│  Text Input     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Text Conditioner (LUTConditioner)                       │
│  - SentencePiece Tokenizer                               │
│  - Embedding Lookup Table (1024 dim, 4000 bins)          │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  FlowLM Model (Transformer + Flow Network)               │
│  - 6-layer Streaming Transformer (1024 dim, 16 heads)    │
│  - AdaLN-conditioned MLP for flow prediction             │
│  - Rotary Position Embeddings (RoPE)                     │
│  - Generates latents autoregressively @ 12.5 Hz          │
└────────┬────────────────────────────────────────────────┘
         │
         ▼ (32-dim latents)
┌─────────────────────────────────────────────────────────┐
│  Mimi Neural Audio Codec                                 │
│  - SEANet Encoder/Decoder (5-layer CNN)                 │
│  - 2-layer Projection Transformers                       │
│  - Dummy Quantizer (32-dimensional latent space)         │
│  - Decodes latents to 24kHz waveform                     │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Audio Output   │
│  (24kHz WAV)    │
└─────────────────┘
```

### Component Details

#### 1. Text Conditioner (`LUTConditioner`)

**Location**: `pocket_tts/conditioners/text.py`

```yaml
Configuration (from b6369a24.yaml):
  dim: 1024              # Output embedding dimension
  n_bins: 4000           # Vocabulary size
  tokenizer: sentencepiece
  tokenizer_path: hf://kyutai/pocket-tts/tokenizer.model
```

**Functionality**:
- Tokenizes input text using SentencePiece tokenizer
- Converts tokens to embeddings via lookup table
- Outputs 1024-dimensional embeddings
- Handles text preprocessing and formatting

**Key Implementation**:
```python
# From flow_lm.py
conditioner = LUTConditioner(
    n_bins=4000,
    tokenizer_path="hf://kyutai/pocket-tts/tokenizer.model",
    dim=1024,
    output_dim=1024
)
```

#### 2. FlowLM Model

**Location**: `pocket_tts/models/flow_lm.py`

The FlowLM is a transformer-based flow language model using Lagrangian Self Distillation (LSD).

**Architecture**:

```yaml
Transformer Configuration:
  d_model: 1024           # Model dimension
  num_layers: 6           # Number of transformer layers
  num_heads: 16           # Number of attention heads
  hidden_scale: 4         # FFN scaling factor (1024 * 4 = 4096)
  max_period: 10000       # RoPE maximum period

Flow Network Configuration:
  depth: 6                # Depth of flow MLP
  dim: 512                # Flow network dimension
```

**Key Components**:

1. **StreamingTransformer**: Custom transformer implementation with:
   - `StreamingMultiheadAttention` with KV caching
   - Rotary Position Embeddings (RoPE) for positional information
   - Stateful modules for streaming generation

2. **SimpleMLPAdaLN**: Adaptive Layer Normalization MLP that predicts flow directions
   - Takes transformer output and time step (t) as input
   - Predicts flow direction for LSD decoding

3. **LSD Decoding**: `lsd_decode()` function implements Lagrangian Self Distillation
   ```python
   def lsd_decode(v_t, x_0, num_steps=1):
       """Rebuilds data sample using Lagrangian Self Distillation"""
       current = x_0
       for i in range(num_steps):
           s = i / num_steps
           t = (i + 1) / num_steps
           flow_dir = v_t(s, t, current)
           current += flow_dir / num_steps
       return current
   ```

**Generation Process**:
1. Initialize with noise (sampled from N(0, temp))
2. Iteratively apply flow network predictions for `lsd_decode_steps` iterations
3. Output 32-dimensional latent vectors at 12.5 Hz (80ms per frame)

#### 3. Mimi Neural Audio Codec

**Location**: `pocket_tts/models/mimi.py`

Mimi is a neural audio codec that compresses/decompresses audio to/from latent representations.

**Configuration**:

```yaml
Audio Configuration:
  sample_rate: 24000      # 24 kHz output
  channels: 1             # Mono audio
  frame_rate: 12.5        # 12.5 Hz latent frame rate

SEANet Encoder/Decoder:
  dimension: 512
  n_filters: 64
  n_residual_layers: 1
  ratios: [6, 5, 4]       # Total downsampling: 6*5*4 = 120
  kernel_size: 7

Projection Transformers:
  d_model: 512
  num_heads: 8
  num_layers: 2
  context: 250            # Context window for attention
  dim_feedforward: 2048

Quantizer:
  dimension: 32           # 32-dimensional latent space
  output_dimension: 512
```

**Architecture**:

**Encoder Path**:
```
Audio (24kHz) → SEANet Encoder → Projection Transformer → Quantizer → Latents (32-dim @ 12.5Hz)
```

1. **SEANet Encoder**: 5-layer convolutional encoder
   - Downsampling ratios: [6, 5, 4] = 120x total
   - 24kHz → 200Hz (after 120x downsampling)
   - Outputs 512-dimensional features

2. **Encoder Transformer**: 2-layer transformer with 8 heads
   - Projects features to 512 dimensions
   - Context window of 250 frames

3. **Quantizer**: Maps to 32-dimensional latent space
   - Compression ratio: 48x (24kHz → 12.5Hz)

**Decoder Path** (reverse of encoder):
```
Latents (32-dim) → Projection Transformer → SEANet Decoder → Audio (24kHz)
```

#### 4. Voice Cloning Pipeline

**Location**: `pocket_tts/models/tts_model.py:get_state_for_audio_prompt()`

Voice cloning creates a model state conditioned on a reference audio clip.

**Process**:

1. **Audio Input**: Reference audio file (3-30 seconds recommended)
2. **Resampling**: Convert to 24kHz if needed
3. **Encoding**:
   ```python
   encoded = self.mimi.encode_to_latent(audio)  # Audio → latents
   latents = encoded.transpose(-1, -2).to(torch.float32)
   conditioning = F.linear(latents, self.flow_lm.speaker_proj_weight)  # 32 → 64 dim
   ```
4. **State Initialization**: Create model state with audio conditioning
5. **State Usage**: Pass state to `generate_audio()` for voice-consistent synthesis

**Key Implementation Details**:
- Uses LRU cache (`@lru_cache(maxsize=2)`) to avoid reprocessing audio prompts
- Supports HuggingFace URLs, local paths, and pre-loaded tensors
- Audio truncated to 30 seconds if needed to prevent memory issues
- Speaker projection weight: `(1024, 512)` matrix for latent space projection

#### 5. Streaming Generation Pipeline

**Location**: `pocket_tts/models/tts_model.py:generate_audio_stream()`

The streaming architecture uses **multithreading** for parallel generation and decoding:

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Thread                               │
│  1. Split long text into sentences/chunks                   │
│  2. For each chunk:                                          │
│     - Prepare text (tokenization, formatting)               │
│     - Start decoder thread                                  │
│     - Start generation thread                               │
└────────┬────────────────────────────────────────────────────┘
         │
         ├─────────────────────────────────────────────────┐
         │                                                 │
         ▼                                                 ▼
┌─────────────────────┐                           ┌─────────────────────┐
│  Generation Thread  │                           │   Decoder Thread    │
│  - Runs FlowLM      │                           │  - Runs Mimi Decoder│
│  - Generates latents│◄── Queue ───────────────► │  - Decodes to audio │
│  @ 12.5 Hz          │   latents_queue          │  - Streams chunks   │
└─────────────────────┘                           └─────────────────────┘
         │                                                 │
         └─────────────────────────────────────────────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │  Audio Chunks   │
                          │  (streamed)     │
                          └─────────────────┘
```

**Thread Responsibilities**:

1. **Generation Thread** (`_autoregressive_generation()`):
   - Runs FlowLM autoregressively
   - Generates one latent per frame (80ms)
   - Puts latents into `latents_queue`
   - Signals completion with `None` sentinel

2. **Decoder Thread** (`_decode_audio_worker()`):
   - Reads latents from `latents_queue`
   - Denormalizes: `latent * emb_std + emb_mean`
   - Transposes and quantizes
   - Decodes via Mimi decoder
   - Puts audio chunks into `result_queue`

3. **Main Thread**:
   - Yields audio chunks from `result_queue` as they become available
   - Enables real-time streaming/playback

**Performance Benefits**:
- **Parallel Processing**: Generation and decoding happen simultaneously
- **Low Latency**: First audio chunk available in ~200ms
- **Memory Efficiency**: Only keeps current frame in memory
- **Real-Time Capable**: Can stream audio as it's generated

### Configuration System

**Location**: `pocket_tts/config/b6369a24.yaml`

```yaml
# Model variant signature
sig: b6369a24

# Weights paths (HuggingFace)
weights_path: hf://kyutai/pocket-tts/tts_b6369a24.safetensors
weights_path_without_voice_cloning: hf://kyutai/pocket-tts-without-voice-cloning/tts_b6369a24.safetensors

# FlowLM Configuration
flow_lm:
  dtype: float32
  flow:
    depth: 6
    dim: 512
  transformer:
    d_model: 1024
    hidden_scale: 4
    max_period: 10000
    num_heads: 16
    num_layers: 6
  lookup_table:
    dim: 1024
    n_bins: 4000
    tokenizer: sentencepiece
    tokenizer_path: hf://kyutai/pocket-tts/tokenizer.model

# Mimi Codec Configuration
mimi:
  dtype: float32
  sample_rate: 24000
  channels: 1
  frame_rate: 12.5
  seanet:
    dimension: 512
    # ... (see full config above)
```

---

## Python API Usage

The Python API provides a simple interface for integrating TTS into your applications.

### Installation

```bash
# Using pip
pip install pocket-tts

# Using uv (recommended)
uv add pocket-tts
```

### Quick Start

```python
from pocket_tts import TTSModel
import scipy.io.wavfile

# Load the model (downloads weights on first use)
tts_model = TTSModel.load_model()

# Get voice state from an audio file
voice_state = tts_model.get_state_for_audio_prompt(
    "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
)

# Generate audio
audio = tts_model.generate_audio(voice_state, "Hello world, this is a test.")

# Save to file
scipy.io.wavfile.write("output.wav", tts_model.sample_rate, audio.numpy())
```

### Core API: TTSModel

#### Class Method: `load_model()`

Load a pre-trained TTS model with specified configuration.

**Signature**:
```python
@classmethod
def load_model(
    variant: str = "b6369a24",
    temp: float = 0.7,
    lsd_decode_steps: int = 1,
    noise_clamp: float | None = None,
    eos_threshold: float = -4.0
) -> TTSModel
```

**Parameters**:
- `variant` (str): Model variant identifier (default: "b6369a24")
- `temp` (float): Sampling temperature (default: 0.7)
  - Higher values = more diverse, potentially lower quality
  - Lower values = more deterministic, conservative
- `lsd_decode_steps` (int): Number of LSD decoding steps (default: 1)
  - More steps = higher quality, slower
  - Fewer steps = faster, lower quality
- `noise_clamp` (float | None): Maximum value for noise sampling (default: None)
  - Prevents extreme values in generation
- `eos_threshold` (float): EOS detection threshold (default: -4.0)
  - Higher values = model continues longer
  - Lower values = model stops earlier

**Returns**: Initialized `TTSModel` instance on CPU

**Example**:
```python
# Default settings
model = TTSModel.load_model()

# High quality
model = TTSModel.load_model(
    temp=0.5,
    lsd_decode_steps=5,
    eos_threshold=-3.0
)

# Fast generation
model = TTSModel.load_model(
    temp=1.0,
    lsd_decode_steps=1
)
```

#### Properties

**`device`** (str)
Returns the device type ("cpu" or "cuda").

```python
model = TTSModel.load_model()
print(f"Model running on: {model.device}")  # "cpu"
```

**`sample_rate`** (int)
Returns the audio sample rate (typically 24000 Hz).

```python
model = TTSModel.load_model()
print(f"Sample rate: {model.sample_rate} Hz")  # 24000
```

#### Method: `get_state_for_audio_prompt()`

Extract model state for a given audio file (voice cloning).

**Signature**:
```python
def get_state_for_audio_prompt(
    audio_conditioning: Path | str | torch.Tensor,
    truncate: bool = False
) -> dict
```

**Parameters**:
- `audio_conditioning`: Audio prompt (voice to clone)
  - Path: Local file path
  - str: URL (HuggingFace, HTTP)
  - torch.Tensor: Pre-loaded audio tensor [channels, samples]
- `truncate` (bool): Truncate audio to 30 seconds (default: False)

**Returns**: Model state dictionary with hidden states and positional information

**Examples**:
```python
model = TTSModel.load_model()

# From HuggingFace URL
voice_state = model.get_state_for_audio_prompt(
    "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
)

# From local file
voice_state = model.get_state_for_audio_prompt("./my_voice.wav")

# From HTTP URL
voice_state = model.get_state_for_audio_prompt(
    "https://huggingface.co/kyutai/tts-voices/resolve/main/..."
)

# With truncation (for long audio)
voice_state = model.get_state_for_audio_prompt(
    "./long_audio.wav",
    truncate=True  # Use first 30 seconds only
)
```

#### Method: `generate_audio()`

Generate complete audio tensor from text input.

**Signature**:
```python
def generate_audio(
    model_state: dict,
    text_to_generate: str,
    frames_after_eos: int | None = None,
    copy_state: bool = True
) -> torch.Tensor
```

**Parameters**:
- `model_state`: Model state from `get_state_for_audio_prompt()`
- `text_to_generate`: Text to convert to speech
- `frames_after_eos` (int | None): Frames to generate after EOS detection
  - None = auto-calculated based on text length
  - Each frame = 80ms
- `copy_state` (bool): Whether to copy the state (default: True)
  - True = preserves original state for reuse
  - False = modifies state in-place

**Returns**: Audio tensor [samples] at model's sample rate

**Example**:
```python
model = TTSModel.load_model()
voice_state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav")

# Generate audio
audio = model.generate_audio(
    voice_state,
    "Hello world!",
    frames_after_eos=2,
    copy_state=True
)

print(f"Generated: {audio.shape[-1]} samples")
print(f"Duration: {audio.shape[-1] / model.sample_rate:.2f} seconds")
```

#### Method: `generate_audio_stream()`

Generate audio streaming chunks from text input.

**Signature**:
```python
def generate_audio_stream(
    model_state: dict,
    text_to_generate: str,
    frames_after_eos: int | None = None,
    copy_state: bool = True
)
```

**Yields**: Audio chunks [samples] as they become available

**Example**:
```python
model = TTSModel.load_model()
voice_state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav")

# Stream generation
for chunk in model.generate_audio_stream(voice_state, "Long text content..."):
    # Process each chunk as it's generated
    print(f"Generated chunk: {chunk.shape[0]} samples")
    # Could save chunks, play in real-time, etc.
```

### Advanced Usage Patterns

#### Voice Management

Preload and reuse multiple voice states for efficiency:

```python
from pocket_tts import TTSModel

model = TTSModel.load_model()

# Preload multiple voices
voices = {
    "casual": model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav"),
    "funny": model.get_state_for_audio_prompt(
        "https://huggingface.co/kyutai/tts-voices/resolve/main/expresso/..."
    ),
    "narrator": model.get_state_for_audio_prompt("./narrator_voice.wav")
}

# Generate with different voices
casual_audio = model.generate_audio(voices["casual"], "Hey there!")
funny_audio = model.generate_audio(voices["funny"], "Good morning.")
narrator_audio = model.generate_audio(voices["narrator"], "Chapter 1...")
```

`★ Insight ─────────────────────────────────────`
**Voice State Caching**:
- The `get_state_for_audio_prompt()` method is wrapped with `@lru_cache(maxsize=2)`
- This means the last 2 audio prompts are automatically cached
- Reusing the same voice state avoids re-encoding the audio
- For production, pre-load voice states at startup and reuse them
- Each voice state is ~10-20MB in memory depending on audio length
`─────────────────────────────────────────────────`

#### Batch Processing

Process multiple texts efficiently by reusing voice state:

```python
from pocket_tts import TTSModel
import scipy.io.wavfile
import torch

model = TTSModel.load_model()
voice_state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav")

# Process multiple texts
texts = [
    "First sentence.",
    "Second sentence.",
    "Third sentence.",
]

audios = []
for text in texts:
    audio = model.generate_audio(voice_state, text)
    audios.append(audio)

# Concatenate all audio
full_audio = torch.cat(audios, dim=0)
scipy.io.wavfile.write("batch_output.wav", model.sample_rate, full_audio.numpy())
```

#### Real-Time Streaming

Stream audio for real-time playback:

```python
from pocket_tts import TTSModel
import pyaudio  # For audio playback

model = TTSModel.load_model()
voice_state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav")

# Initialize audio output
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=model.sample_rate,
    output=True
)

# Stream and play in real-time
for chunk in model.generate_audio_stream(voice_state, "This is a long text that will be streamed..."):
    # Convert to float32 and play
    audio_float32 = chunk.float().numpy()
    stream.write(audio_float32.tobytes())

stream.stop_stream()
stream.close()
p.terminate()
```

#### Long Text Handling

The model automatically splits long texts into chunks:

```python
model = TTSModel.load_model()
voice_state = model.get_state_for_audio_prompt("hf://kyutai/tts-voices/alba-mackenna/casual.wav")

# Long text is automatically split into sentences
long_text = """
This is a very long text that spans multiple sentences.
The model will automatically split it into chunks of up to 50 tokens each.
This ensures smooth generation and prevents memory issues.
Each sentence boundary is used as a natural splitting point.
"""

# Works seamlessly for long texts
audio = model.generate_audio(voice_state, long_text)
```

---

## Voice Cloning Implementation

Voice cloning is one of Pocket TTS's most powerful features, allowing you to clone voices from short audio clips.

### How Voice Cloning Works

`★ Insight ─────────────────────────────────────`
**Voice Cloning Pipeline**:
1. **Audio Encoding**: Reference audio is encoded by Mimi encoder into latent representations
2. **Feature Extraction**: Latents are projected from 32-dim to 64-dim space via speaker projection matrix
3. **State Conditioning**: These features are used to initialize the model's internal state
4. **Generation**: When generating speech, the transformer attention mechanism references this state, ensuring the output matches the voice characteristics

The key innovation is that voice characteristics are captured in the **model state** rather than explicit speaker embeddings, enabling zero-shot voice cloning.
`─────────────────────────────────────────────────`

### Technical Implementation

**Location**: `pocket_tts/models/tts_model.py:get_state_for_audio_prompt()`

**Step-by-Step Process**:

```python
@torch.no_grad
def get_state_for_audio_prompt(
    self,
    audio_conditioning: Path | str | torch.Tensor,
    truncate: bool = False
) -> dict:
    # 1. Load or validate audio
    if isinstance(audio_conditioning, str) and audio_conditioning in PREDEFINED_VOICES:
        prompt = load_predefined_voice(audio_conditioning)
    else:
        if isinstance(audio_conditioning, str):
            audio_conditioning = download_if_necessary(audio_conditioning)

        if isinstance(audio_conditioning, Path):
            audio, sr = audio_read(audio_conditioning)
            if truncate:
                max_samples = int(30 * sr)  # 30 seconds
                audio = audio[..., :max_samples]

            audio_conditioning = convert_audio(
                audio, sr, self.config.mimi.sample_rate, 1
            )

    # 2. Encode audio to latents
    encoded = self.mimi.encode_to_latent(audio_conditioning.unsqueeze(0))

    # 3. Transpose and project to flow model space
    latents = encoded.transpose(-1, -2).to(torch.float32)
    conditioning = F.linear(latents, self.flow_lm.speaker_proj_weight)

    # 4. Initialize model state
    model_state = init_states(self.flow_lm, batch_size=1, sequence_length=1000)

    # 5. Prompt the model with audio conditioning
    self._run_flow_lm_and_increment_step(
        model_state=model_state,
        audio_conditioning=conditioning
    )

    return model_state
```

### Predefined Voices

Pocket TTS includes a catalog of pre-packaged voices:

```python
PREDEFINED_VOICES = {
    "alba": "hf://kyutai/tts-voices/alba-mackenna/casual.wav",
    "marius": "hf://kyutai/tts-voices/voice-donations/Selfie.wav",
    "javert": "hf://kyutai/tts-voices/voice-donations/Butter.wav",
    "jean": "hf://kyutai/tts-voices/ears/p010/freeform_speech_01.wav",
    "fantine": "hf://kyutai/tts-voices/vctk/p244_023.wav",
    "cosette": "hf://kyutai/tts-voices/expresso/ex04-ex02_confused_001_channel1_499s.wav",
    "eponine": "hf://kyutai/tts-voices/vctk/p262_023.wav",
    "azelma": "hf://kyutai/tts-voices/vctk/p303_023.wav",
}
```

**Usage**:
```python
model = TTSModel.load_model()

# Use predefined voice by name
voice_state = model.get_state_for_audio_prompt("alba")
```

### Custom Voice Cloning

Clone a custom voice from your own audio files:

```python
from pocket_tts import TTSModel

model = TTSModel.load_model()

# Clone from local file
voice_state = model.get_state_for_audio_prompt("./my_recording.wav")

# Clone from URL
voice_state = model.get_state_for_audio_prompt(
    "https://example.com/voice_sample.wav"
)

# Generate speech with cloned voice
audio = model.generate_audio(voice_state, "This is my cloned voice speaking!")
```

### Best Practices for Voice Cloning

#### Audio Quality Requirements

**Optimal Audio Characteristics**:
- **Duration**: 3-10 seconds (minimum 3 seconds, max 30 seconds)
- **Sample Rate**: Any (automatically resampled to 24kHz)
- **Format**: WAV, MP3, or other audio formats
- **Content**: Clear speech with minimal background noise
- **Speaker**: Single speaker, consistent voice

**Dos and Don'ts**:

✅ **DO**:
- Use clear, recorded speech
- Use consistent voice samples
- Trim silence from beginning/end
- Use 16-bit or higher audio quality
- Test with multiple samples

❌ **DON'T**:
- Use music or non-speech audio
- Use multiple speakers in one sample
- Use very short clips (< 2 seconds)
- Use heavily compressed audio
- Use low-quality recordings (8-bit, < 8kHz)

#### Audio Preparation

```python
from pocket_tts import TTSModel
import scipy.io.wavfile

model = TTSModel.load_model()

# Load and preprocess audio
sample_rate, audio = scipy.io.wavfile.read("recording.wav")

# If stereo, convert to mono
if audio.ndim == 2:
    audio = audio.mean(axis=1)

# Normalize audio
audio = audio.astype(float)
audio = audio / audio.max()

# Trim silence (simple energy-based threshold)
threshold = 0.01
above_threshold = audio > threshold
if above_threshold.any():
    start = above_threshold.argmax()
    end = len(audio) - above_threshold[::-1].argmax()
    audio = audio[start:end]

# Save cleaned audio
scipy.io.wavfile.write("cleaned_voice.wav", sample_rate, audio.astype("float32"))

# Now use for cloning
voice_state = model.get_state_for_audio_prompt("cleaned_voice.wav")
```

#### Voice Quality Assessment

```python
import torch

def assess_voice_quality(audio_tensor):
    """Assess audio quality for voice cloning"""
    # Check duration
    duration_sec = audio_tensor.shape[-1] / 24000
    if duration_sec < 3:
        return "Too short (< 3s)"
    elif duration_sec > 30:
        return "Too long (> 30s), will be truncated"

    # Check for clipping
    if audio_tensor.abs().max() > 0.99:
        return "Warning: Audio may be clipped"

    # Check dynamic range
    dynamic_range = 20 * torch.log10(audio_tensor.abs().max() / (audio_tensor.abs().mean() + 1e-8))
    if dynamic_range < 10:
        return "Low dynamic range"

    return "Good quality"

# Usage
model = TTSModel.load_model()
# Assuming you have audio_tensor loaded
quality = assess_voice_quality(audio_tensor)
print(f"Voice quality: {quality}")
```

### HuggingFace Integration

For full voice cloning capabilities, you need to accept the model terms on HuggingFace:

```bash
# Login to HuggingFace CLI
uvx hf login

# Or install huggingface-cli
pip install huggingface_hub
huggingface-cli login
```

Then accept the terms at: https://huggingface.co/kyutai/pocket-tts

**Without HuggingFace Access**:
- You can still use the 8 predefined voices
- Custom voice cloning requires authentication
- Use `weights_path_without_voice_cloning` model variant

---

## Setup and Installation

### Prerequisites

- **Python**: 3.10, 3.11, 3.12, 3.13, or 3.14
- **PyTorch**: 2.5 or later (CPU version recommended)
- **Operating System**: Linux, macOS, or Windows

### Installation Methods

#### Method 1: pip (Standard)

```bash
# Install with pip
pip install pocket-tts

# Verify installation
python -c "from pocket_tts import TTSModel; print(TTSModel.load_model())"
```

#### Method 2: uv (Recommended)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to your project
uv add pocket-tts

# Or run directly without installation
uvx pocket-tts generate
```

**Advantages of uv**:
- Faster dependency resolution
- Isolated environments
- Automatic CPU-only PyTorch installation
- Better caching

#### Method 3: Development Installation

```bash
# Clone repository
git clone https://github.com/kyutai-labs/pocket-tts.git
cd pocket-tts

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install in editable mode
uv sync

# Install pre-commit hooks
uvx pre-commit install
```

### Verification

Test your installation:

```bash
# CLI test
uvx pocket-tts generate --text "Installation successful!"

# Python API test
python -c "
from pocket_tts import TTSModel
model = TTSModel.load_model()
print('Model loaded successfully!')
print(f'Device: {model.device}')
print(f'Sample rate: {model.sample_rate}')
"
```

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| CPU | 2 cores | 4+ cores |
| Storage | 500 MB | 1 GB |
| Python | 3.10 | 3.11 or 3.12 |

### Docker Installation

```dockerfile
# From Dockerfile in repository
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY . .

# Install dependencies
RUN uv sync --frozen

EXPOSE 8000

# Run server
CMD ["uv", "run", "pocket-tts", "serve", "--host", "0.0.0.0"]
```

**Build and run**:
```bash
docker build -t pocket-tts .
docker run -p 8000:8000 pocket-tts
```

### Troubleshooting Installation

#### PyTorch Version Issues

```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# If < 2.5, upgrade
pip install --upgrade torch

# For CPU-only PyTorch (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### HuggingFace Authentication

```bash
# Install huggingface-cli
pip install huggingface_hub

# Login
huggingface-cli login

# Accept terms at: https://huggingface.co/kyutai/pocket-tts
```

#### Download Issues

If downloads fail, weights are cached locally:

```bash
# Linux/macOS
ls -la ~/.cache/huggingface/hub/

# Set custom cache directory
export HF_HOME=/custom/path
```

---

## CLI Usage

Pocket TTS provides two main CLI commands: `generate` and `serve`.

### The `generate` Command

Generate speech from text directly in the terminal.

#### Basic Usage

```bash
# Using uv (recommended)
uvx pocket-tts generate

# Or if installed with pip
pocket-tts generate
```

**Output**: Creates `./tts_output.wav` with default text and voice.

#### Command Options

**Core Options**:
```bash
--text TEXT                    # Text to generate (default: "Hello world!...")
--voice VOICE                  # Audio conditioning file or URL
--output-path OUTPUT_PATH      # Output WAV file path (default: ./tts_output.wav)
```

**Generation Parameters**:
```bash
--variant VARIANT              # Model signature (default: b6369a24)
--lsd-decode-steps N           # Generation steps (default: 1)
--temperature TEMP             # Temperature 0.0-1.0 (default: 0.7)
--noise-clamp VALUE            # Noise clamp value (default: None)
--eos-threshold VALUE          # EOS threshold (default: -4.0)
--frames-after-eos N           # Frames after EOS (default: auto)
```

**Performance Options**:
```bash
--device DEVICE                # Device: cpu or cuda (default: cpu)
--quiet, -q                    # Disable logging output
```

#### Examples

**Basic Generation**:
```bash
# Default
pocket-tts generate

# Custom text
pocket-tts generate --text "Hello, this is a custom message."

# Custom output path
pocket-tts generate --output-path "./output.wav"

# Multiple texts
pocket-tts generate --text "First message." --output-path "first.wav"
pocket-tts generate --text "Second message." --output-path "second.wav"
```

**Voice Selection**:
```bash
# Use predefined voice from HuggingFace
pocket-tts generate --voice "hf://kyutai/tts-voices/alba-mackenna/casual.wav"

# Use local voice file
pocket-tts generate --voice "./my_voice.wav"

# Use voice from URL
pocket-tts generate --voice "https://example.com/voice.wav"
```

**Quality Tuning**:
```bash
# High quality (more steps, lower temperature)
pocket-tts generate \
  --lsd-decode-steps 5 \
  --temperature 0.5 \
  --output-path "high_quality.wav"

# Fast generation (fewer steps)
pocket-tts generate \
  --lsd-decode-steps 1 \
  --output-path "fast.wav"

# More expressive (higher temperature)
pocket-tts generate \
  --temperature 1.0 \
  --output-path "expressive.wav"

# Longer generation after EOS
pocket-tts generate \
  --frames-after-eos 5 \
  --output-path "extended.wav"
```

**Batch Generation**:
```bash
# Generate multiple files
for text in "Hello" "World" "Test"; do
  pocket-tts generate --text "$text" --output-path "${text}.wav"
done
```

### The `serve` Command

Start a FastAPI web server with both web interface and HTTP API.

#### Basic Usage

```bash
# Using uv
uvx pocket-tts serve

# Or if installed
pocket-tts serve
```

**Access**: Navigate to `http://localhost:8000`

#### Command Options

```bash
--voice VOICE                  # Default voice for server
--host HOST                    # Host to bind (default: localhost)
--port PORT                    # Port to bind (default: 8000)
--reload                       # Enable auto-reload for development
```

#### Examples

```bash
# Default server
pocket-tts serve

# Custom host and port
pocket-tts serve --host "0.0.0.0" --port 8080

# Custom default voice
pocket-tts serve --voice "hf://kyutai/tts-voices/alba-mackenna/casual.wav"

# Development mode with auto-reload
pocket-tts serve --reload
```

#### Web Interface

The web interface provides:
- **Text Input**: Enter text to synthesize
- **Voice Selection**: Choose from predefined voices or upload custom
- **Real-time Generation**: Audio plays as it's generated
- **Download**: Save generated audio files

#### HTTP API

The server exposes a REST API:

```python
import requests

# Server URL
url = "http://localhost:8000"

# Generate speech
response = requests.post(
    f"{url}/generate",
    json={
        "text": "Hello from the API!",
        "voice": "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
    }
)

# Save audio
with open("api_output.wav", "wb") as f:
    f.write(response.content)
```

**API Endpoints**:
- `GET /`: Web interface
- `POST /generate`: Generate speech from text
- `GET /voices`: List available voices

`★ Insight ─────────────────────────────────────`
**Server vs CLI Performance**:
- **CLI Mode**: Model loads and unloads for each generation (~2-3s overhead)
- **Server Mode**: Model stays in memory between requests (no overhead)
- **Recommendation**: Use server mode for testing multiple voices/texts
- **Parallel Processing**: Server is NOT thread-safe; processes requests sequentially
`─────────────────────────────────────────────────`

---

## Advanced Configuration

### Generation Parameters

Understanding the key parameters for quality and speed tuning.

#### Temperature (`temp`)

Controls randomness in generation.

```python
# Range: 0.0 to 1.0+
model = TTSModel.load_model(temp=0.7)
```

**Values**:
- `0.1-0.3`: Very deterministic, monotone
- `0.5-0.7`: Balanced (recommended)
- `0.8-1.0`: More expressive, varied
- `1.0+`: Very creative, potentially unstable

**Effect**:
- Lower = more conservative, repeats patterns
- Higher = more varied, but may introduce artifacts

#### LSD Decode Steps (`lsd_decode_steps`)

Number of flow matching steps.

```python
# Range: 1-10+
model = TTSModel.load_model(lsd_decode_steps=5)
```

**Values**:
- `1`: Fastest, good quality
- `3-5`: Higher quality, slower
- `10+`: Best quality, much slower

**Performance Impact**:
```
1 step:  ~6x real-time (fastest)
5 steps: ~2x real-time
10 steps: ~1x real-time (slowest)
```

#### EOS Threshold (`eos_threshold`)

End-of-speech detection threshold.

```python
# Range: -10.0 to 0.0
model = TTSModel.load_model(eos_threshold=-4.0)
```

**Values**:
- `-3.0`: Stops earlier (may cut off)
- `-4.0`: Balanced (recommended)
- `-5.0`: Continues longer (may add silence)

#### Noise Clamp (`noise_clamp`)

Limits extreme values in noise sampling.

```python
model = TTSModel.load_model(noise_clamp=2.0)
```

**Purpose**: Prevents extreme values that may cause audio artifacts

### Quality vs Speed Trade-offs

```python
from pocket_tts import TTSModel

# Fastest (6x real-time)
fast_model = TTSModel.load_model(
    temp=0.7,
    lsd_decode_steps=1
)

# Balanced (3x real-time)
balanced_model = TTSModel.load_model(
    temp=0.6,
    lsd_decode_steps=3
)

# Highest quality (1x real-time)
quality_model = TTSModel.load_model(
    temp=0.5,
    lsd_decode_steps=10,
    noise_clamp=2.0
)
```

### Custom Configuration

You can create custom model configurations by editing YAML files:

```yaml
# pocket_tts/config/my_config.yaml
weights_path: hf://kyutai/pocket-tts/tts_b6369a24.safetensors

flow_lm:
  dtype: float32
  transformer:
    d_model: 1024
    num_layers: 6
    num_heads: 16
  # ... (see full config in b6369a24.yaml)

mimi:
  sample_rate: 24000
  # ... (see full config in b6369a24.yaml)
```

**Load custom config**:
```python
from pocket_tts import TTSModel
from pocket_tts.utils.config import load_config

# Load custom config
config = load_config("path/to/my_config.yaml")

# Initialize model with custom config
model = TTSModel._from_pydantic_config_with_weights(
    config,
    temp=0.7,
    lsd_decode_steps=1,
    noise_clamp=None,
    eos_threshold=-4.0
)
```

---

## Development Guide

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/kyutai-labs/pocket-tts.git
cd pocket-tts

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install pre-commit hooks
uvx pre-commit install
```

### Running Tests

```bash
# Run all tests with 3 parallel workers
uv run pytest -n 3 -v

# Run specific test file
uv run pytest tests/test_python_api.py -v

# Run specific test
uv run pytest tests/test_python_api.py::test_generate_audio -v
```

### Code Quality

Pre-commit hooks handle formatting automatically:
- **Ruff**: Linting and formatting
- **Import sorting**: Automatically sorts imports
- **Line length**: 100 characters
- **Line endings**: LF (Unix-style)

**Manual checks**:
```bash
# Run all pre-commit hooks
uvx pre-commit run --all-files

# Run ruff only
uv run ruff check .
uv run ruff format .
```

### Project Structure

```
pocket_tts/
├── __init__.py              # Public API (exports TTSModel)
├── __main__.py              # Python module entry point
├── main.py                  # CLI implementation
├── conditioners/            # Text conditioning
│   ├── base.py             # TokenizedText class
│   └── text.py             # LUTConditioner
├── config/                  # Model configurations
│   └── b6369a24.yaml       # Default config
├── data/                    # Audio utilities
│   ├── audio.py            # Audio I/O
│   └── audio_utils.py      # Audio processing
├── models/                  # Core models
│   ├── flow_lm.py          # Flow language model
│   ├── mimi.py             # Mimi codec
│   └── tts_model.py        # Main TTS model
├── modules/                 # Neural network modules
│   ├── transformer.py      # Streaming transformers
│   ├── mlp.py              # AdaLN MLP
│   ├── seanet.py           # SEANet encoder/decoder
│   └── ...
├── utils/                   # Utilities
│   ├── config.py           # Pydantic configs
│   ├── utils.py            # HF downloads, timing
│   └── weights_loading.py  # Weight loading
└── static/                  # Web interface
    └── index.html
```

### Common Development Tasks

#### Adding a New Predefined Voice

```python
# In pocket_tts/utils/utils.py
PREDEFINED_VOICES = {
    # ... existing voices
    "new_voice": "hf://kyutai/tts-voices/new-speaker/style.wav",
}
```

#### Modifying Model Architecture

1. Update config YAML file
2. Update Pydantic config classes in `utils/config.py`
3. Modify model class in `models/`
4. Update weight loading in `utils/weights_loading.py`

#### Adding CLI Options

Edit `pocket_tts/main.py`:

```python
import typer

app = typer.Typer()

@app.command()
def generate(
    new_option: str = typer.Option(default, help="Description"),
    # ... existing options
):
    # Implementation
    pass
```

### Testing

**Test files**:
- `tests/test_python_api.py`: Public API tests
- `tests/test_cli_generate.py`: CLI tests
- `tests/test_documentation_examples.py`: Doc example tests

**Example test**:
```python
import pytest
from pocket_tts import TTSModel

def test_generate_audio():
    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt("alba")
    audio = model.generate_audio(voice_state, "Test.")
    assert audio.shape[-1] > 0
    assert audio.shape[-1] / model.sample_rate > 0
```

---

## Performance Characteristics

`★ Insight ─────────────────────────────────────`
**Performance Optimization Techniques**:
- **CPU Threading**: Model uses `torch.set_num_threads(1)` to avoid oversubscription
- **KV Caching**: Transformer caches key-value pairs for O(1) attention to previous tokens
- **Dual-Thread Pipeline**: Generation and decoding run in parallel, maximizing CPU utilization
- **Memory Efficiency**: Streaming generation keeps only current frame in memory
- **LRU Caching**: Voice prompts are cached to avoid re-encoding
`─────────────────────────────────────────────────`

### Benchmarks

**Hardware**: MacBook Air M4

| Configuration | Speed | Quality |
|--------------|-------|---------|
| 1 step, temp=0.7 | ~6x real-time | Good |
| 3 steps, temp=0.6 | ~3x real-time | Better |
| 5 steps, temp=0.5 | ~2x real-time | Best |

**Latency**:
- First audio chunk: ~200ms
- Subsequent chunks: ~80ms each

### Memory Usage

| Component | Memory |
|-----------|--------|
| Model weights | ~400 MB |
| Voice state | ~10-20 MB |
| Per-generation overhead | ~50 MB |
| **Total (per model)** | ~470 MB |

### CPU Usage

- **Cores used**: 2 cores (configurable)
- **Utilization**: 80-95% during generation
- **Power efficiency**: Optimized for laptop CPUs

### Optimization Tips

**For Speed**:
```python
model = TTSModel.load_model(
    temp=0.7,
    lsd_decode_steps=1  # Fewest steps
)
```

**For Quality**:
```python
model = TTSModel.load_model(
    temp=0.5,
    lsd_decode_steps=5  # More steps
)
```

**For Batch Processing**:
```python
# Reuse voice state to avoid re-encoding
voice_state = model.get_state_for_audio_prompt("alba")

for text in texts:
    audio = model.generate_audio(voice_state, text, copy_state=False)
```

---

## Troubleshooting

### Common Issues

#### Issue: "PyTorch version 2.4.0 produces incorrect audio"

**Solution**: Upgrade PyTorch to 2.5+
```bash
pip install --upgrade torch
```

#### Issue: "No voice cloning without HuggingFace authentication"

**Solution**:
```bash
# Login to HuggingFace
uvx hf login

# Accept terms at: https://huggingface.co/kyutai/pocket-tts
```

#### Issue: "Generation reached maximum length without EOS"

**Causes**:
- Very long text
- Unusual text patterns
- Incorrect EOS threshold

**Solutions**:
```python
# Increase max length or adjust EOS
audio = model.generate_audio(
    voice_state,
    text,
    frames_after_eos=10  # Increase
)

# Or adjust EOS threshold
model = TTSModel.load_model(eos_threshold=-3.0)
```

#### Issue: "Poor audio quality"

**Solutions**:
```python
# Increase decode steps
model = TTSModel.load_model(lsd_decode_steps=5)

# Adjust temperature
model = TTSModel.load_model(temp=0.5)

# Use noise clamp
model = TTSModel.load_model(noise_clamp=2.0)
```

#### Issue: "Voice cloning doesn't match reference"

**Solutions**:
- Use longer audio clip (5-10 seconds)
- Ensure clear speech in reference
- Remove background noise
- Use consistent voice in reference

#### Issue: "Slow generation"

**Solutions**:
```python
# Reduce decode steps
model = TTSModel.load_model(lsd_decode_steps=1)

# Use server mode for multiple generations
pocket-tts serve  # Keeps model in memory

# Check CPU usage
# - Ensure not running other heavy tasks
# - Verify 2+ cores available
```

### Debug Mode

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

model = TTSModel.load_model()
# ... will show detailed timing information
```

### Getting Help

- **GitHub Issues**: https://github.com/kyutai-labs/pocket-tts/issues
- **Discussions**: https://github.com/kyutai-labs/pocket-tts/discussions
- **Documentation**: https://github.com/kyutai-labs/pocket-tts/tree/main/docs

---

## Appendix

### A. Model Weights

**Download Locations**:
- Full model: `hf://kyutai/pocket-tts/tts_b6369a24.safetensors`
- Without voice cloning: `hf://kyutai/pocket-tts-without-voice-cloning/tts_b6369a24.safetensors`
- Tokenizer: `hf://kyutai/pocket-tts/tokenizer.model`

**Caching**:
- Linux/macOS: `~/.cache/huggingface/hub/`
- Windows: `C:\Users\<username>\.cache\huggingface\hub\`

### B. Voice Catalog

All available predefined voices:

| Name | Path | Description |
|------|------|-------------|
| alba | hf://kyutai/tts-voices/alba-mackenna/casual.wav | Casual female |
| marius | hf://kyutai/tts-voices/voice-donations/Selfie.wav | Male |
| javert | hf://kyutai/tts-voices/voice-donations/Butter.wav | Male |
| jean | hf://kyutai/tts-voices/ears/p010/freeform_speech_01.wav | Male |
| fantine | hf://kyutai/tts-voices/vctk/p244_023.wav | Female |
| cosette | hf://kyutai/tts-voices/expresso/ex04-ex02_confused_001_channel1_499s.wav | Expressive |
| eponine | hf://kyutai/tts-voices/vctk/p262_023.wav | Female |
| azelma | hf://kyutai/tts-voices/vctk/p303_023.wav | Female |

**License Information**: https://huggingface.co/kyutai/tts-voices

### C. API Reference Summary

```python
class TTSModel:
    @classmethod
    def load_model(
        variant: str = "b6369a24",
        temp: float = 0.7,
        lsd_decode_steps: int = 1,
        noise_clamp: float | None = None,
        eos_threshold: float = -4.0
    ) -> TTSModel

    @property
    def device(self) -> str

    @property
    def sample_rate(self) -> int

    def get_state_for_audio_prompt(
        audio_conditioning: Path | str | torch.Tensor,
        truncate: bool = False
    ) -> dict

    def generate_audio(
        model_state: dict,
        text_to_generate: str,
        frames_after_eos: int | None = None,
        copy_state: bool = True
    ) -> torch.Tensor

    def generate_audio_stream(
        model_state: dict,
        text_to_generate: str,
        frames_after_eos: int | None = None,
        copy_state: bool = True
    ) -> Generator[torch.Tensor, None, None]
```

### D. Configuration Reference

**FlowLM Config**:
```yaml
flow_lm:
  dtype: float32
  flow:
    depth: 6
    dim: 512
  transformer:
    d_model: 1024
    hidden_scale: 4
    max_period: 10000
    num_heads: 16
    num_layers: 6
  lookup_table:
    dim: 1024
    n_bins: 4000
    tokenizer: sentencepiece
```

**Mimi Config**:
```yaml
mimi:
  dtype: float32
  sample_rate: 24000
  channels: 1
  frame_rate: 12.5
  seanet:
    dimension: 512
    ratios: [6, 5, 4]
    # ...
```

---

## Conclusion

Pocket TTS represents a significant advancement in CPU-based text-to-speech synthesis, combining:

- **State-of-the-art quality** with voice cloning
- **Efficient architecture** using flow matching and neural codecs
- **Production-ready** Python API and CLI
- **Streaming capabilities** for real-time applications

The model is particularly well-suited for:
- Edge devices and local deployment
- Applications requiring low latency
- Voice cloning and customization
- Real-time streaming applications

For questions, contributions, or support, visit:
- **GitHub**: https://github.com/kyutai-labs/pocket-tts
- **Hugging Face**: https://huggingface.co/kyutai/pocket-tts
- **Demo**: https://kyutai.org/tts

---

*Document Version: 1.0*
*Last Updated: 2025-01-24*
*Generated for Kyutai Labs Pocket TTS*
