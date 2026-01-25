# Generate Command Documentation

The `generate` command allows you to generate speech from text directly from the command line using Kyutai Pocket TTS.

## Basic Usage

```bash
uvx pocket-tts generate
# or if installed manually:
pocket-tts generate
```

This will generate a WAV file `./tts_output.wav` with the default text and voice.

## Command Options

### Core Options

- `--text TEXT`: Text to generate (default: "Hello world! I am Kyutai Pocket TTS. I'm fast enough to run on small CPUs. I hope you'll like me.")
- `--voice VOICE`: Path to audio conditioning file (voice to clone) (default: "hf://kyutai/tts-voices/alba-mackenna/casual.wav"). Urls and local paths are supported.
- `--output-path OUTPUT_PATH`: Output path for generated audio (default: "./tts_output.wav")

### Generation Parameters

- `--variant VARIANT`: Model signature (default: "b6369a24")
- `--lsd-decode-steps LSD_DECODE_STEPS`: Number of generation steps (default: 1)
- `--temperature TEMPERATURE`: Temperature for generation (default: 0.7)
- `--noise-clamp NOISE_CLAMP`: Noise clamp value (default: None)
- `--eos-threshold EOS_THRESHOLD`: EOS threshold (default: -4.0)
- `--frames-after-eos FRAMES_AFTER_EOS`: Number of frames to generate after EOS (default: None, auto-calculated based on the text length). Each frame is 80ms.

### Performance Options

- `--device DEVICE`: Device to use (default: "cpu", you may not get a speedup by using a gpu since it's a small model)
- `--compile`: Enable `torch.compile` for inference (default: off)
- `--compile-backend`: torch.compile backend (default: "inductor")
- `--compile-mode`: torch.compile mode (default: "reduce-overhead")
- `--compile-fullgraph`: torch.compile fullgraph (default: false)
- `--compile-dynamic`: torch.compile dynamic (default: false)
- `--compile-targets`: Compile targets (all, flow-lm, mimi-decoder)
- `--quiet`, `-q`: Disable logging output

## Examples

### Basic Generation

```bash
# Generate with default settings
pocket-tts generate

# Custom text
pocket-tts generate --text "Hello, this is a custom message."

# Custom output path
pocket-tts generate --output-path "./my_audio.wav"
```

### Voice Selection

```bash
# Use different voice from HuggingFace
pocket-tts generate --voice "hf://kyutai/tts-voices/jessica-jian/casual.wav"

# Use local voice file
pocket-tts generate --voice "./my_voice.wav"
```

### Quality Tuning

```bash
# Higher quality (more steps)
pocket-tts generate --lsd-decode-steps 5 --temperature 0.5

# More expressive (higher temperature)
pocket-tts generate --temperature 1.0

# Adjust EOS threshold, smaller means finishing earlier.
pocket-tts generate --eos-threshold -3.0
```

### Advanced Examples

#### Batch Processing

Generate multiple audio files from a text file (one line per audio):

```bash
# Create a file with multiple texts
cat > texts.txt << EOF
First sentence to generate.
Second sentence to generate.
Third sentence to generate.
EOF

# Generate each line as a separate file
i=1
while read -r text; do
  pocket-tts generate --text "$text" --output-path "output_$i.wav"
  i=$((i+1))
done < texts.txt
```

#### Streaming Long Text

For very long texts, use streaming to handle generation efficiently:

```python
from pocket_tts import TTSModel
import scipy.io.wavfile
import torch

model = TTSModel.load_model()
voice_state = model.get_state_for_audio_prompt("alba")

long_text = """Your very long text here...
This can be thousands of characters.
The streaming API will handle it efficiently."""

# Stream generation
all_chunks = []
for chunk in model.generate_audio_stream(voice_state, long_text):
    all_chunks.append(chunk)
    # Process chunk immediately if needed

# Combine chunks
full_audio = torch.cat(all_chunks, dim=0)
scipy.io.wavfile.write("output.wav", model.sample_rate, full_audio.numpy())
```

#### Voice Conversion

Convert text to speech using different voices:

```bash
# Generate the same text with different voices
for voice in alba marius javert jean fantine; do
  pocket-tts generate \
    --text "Hello, this is a test of voice conversion." \
    --voice "$voice" \
    --output-path "output_${voice}.wav"
done
```

#### Optimized Generation

Use compilation for faster repeated generations:

```bash
# First generation will be slower (compilation)
pocket-tts generate --compile --text "First generation"

# Subsequent generations will be faster
pocket-tts generate --compile --text "Second generation"
```

#### High-Quality Generation

For best audio quality (slower generation):

```bash
# High quality settings
pocket-tts generate \
  --temperature 0.8 \
  --lsd-decode-steps 8 \
  --noise-clamp none \
  --text "This will be high quality audio."
```

#### Fast Generation

For quick generation (lower quality):

```bash
# Fast settings
pocket-tts generate \
  --temperature 0.7 \
  --lsd-decode-steps 1 \
  --noise-clamp 1.0 \
  --text "This will be fast."
```

#### Custom Voice from File

Clone a voice from a custom audio file:

```bash
# Record or prepare your voice sample
# Then use it for generation
pocket-tts generate \
  --voice "./my_voice_sample.wav" \
  --text "This will use my custom voice." \
  --output-path "custom_voice_output.wav"
```

## Output Format

The generate command always outputs WAV files in the following format:

- **Sample Rate**: 24kHz
- **Channels**: Mono
- **Bit Depth**: 16-bit PCM
- **Format**: Standard WAV file

For more advanced usage, see the [Python API documentation](python-api.md) or consider using the [serve command](serve.md) for web-based generation and quick iteration.
