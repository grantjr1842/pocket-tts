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

## Output Format

The generate command always outputs WAV files in the following format:

- **Sample Rate**: 24kHz
- **Channels**: Mono
- **Bit Depth**: 16-bit PCM
- **Format**: Standard WAV file

## Advanced Generation Examples

### Batch Processing

```python
from pocket_tts import TTSModel, save_audio
import torch
import json

def batch_generate(texts, voice="alba", output_dir="./batch_output"):
    """Generate multiple texts efficiently."""
    model = TTSModel.load_model()
    model.compile_for_inference()
    
    voice_state = model.get_state_for_audio_prompt(voice)
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for i, text in enumerate(texts, 1):
        try:
            audio = model.generate_audio(voice_state, text, copy_state=True)
            
            filename = os.path.join(output_dir, f"output_{i:03d}.wav")
            save_audio(filename, audio, model.sample_rate)
            
            results.append({"text": text, "file": filename, "success": True})
            print(f"Generated {i}/{len(texts)}: {filename}")
            
        except Exception as e:
            results.append({"text": text, "error": str(e), "success": False})
            print(f"Failed {i}/{len(texts)}: {e}")
    
    # Save results summary
    with open(os.path.join(output_dir, "batch_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Batch complete: {len([r for r in results if r['success']])}/{len(texts)} successful")
    return results

# Usage
texts = [
    "First sentence to generate.",
    "Second sentence to generate.",
    "Third sentence to generate."
]

batch_generate(texts, voice="marius", output_dir="./my_batch")
```

### Voice Conversion and Processing

```python
from pocket_tts import TTSModel, load_wav, save_audio, normalize_audio, apply_fade
import numpy as np
import torch

def process_voice_file(input_voice_path, output_voice_path):
    """Process and enhance a voice file."""
    model = TTSModel.load_model()
    
    try:
        # Load original voice
        original_audio, sr = load_wav(input_voice_path)
        print(f"Loaded voice: {original_audio.shape}, {sr} Hz")
        
        # Process audio
        processed_audio = original_audio.numpy()
        
        # Normalize audio levels
        processed_audio = normalize_audio(processed_audio, gain=1.1)
        print("Applied normalization and gain")
        
        # Apply fade to reduce clicks
        processed_audio = apply_fade(
            processed_audio, 
            fade_in_ms=10, 
            fade_out_ms=10,
            sample_rate=sr
        )
        print("Applied fade in/out")
        
        # Convert back to tensor
        processed_tensor = torch.from_numpy(processed_audio)
        
        # Test the processed voice
        voice_state = model.get_state_for_audio_prompt(processed_tensor)
        test_audio = model.generate_audio(voice_state, "Testing the processed voice.")
        
        # Save processed voice
        save_audio(output_voice_path, processed_tensor, sr)
        save_audio("test_output.wav", test_audio, model.sample_rate)
        
        print(f"Processed voice saved to: {output_voice_path}")
        print(f"Test audio saved to: test_output.wav}")
        
        return True
        
    except Exception as e:
        print(f"Voice processing failed: {e}")
        return False

# Usage
success = process_voice_file("input_voice.wav", "processed_voice.wav")
if success:
    print("Voice processing completed successfully")
```

### Real-time Generation with Caching

```python
from pocket_tts import TTSModel
import asyncio
import time
from collections import OrderedDict

class RealTimeTTS:
    def __init__(self, max_cache_size=10):
        self.model = TTSModel.load_model()
        self.model.compile_for_inference()
        
        # LRU cache for voice states ( OrderedDict for LRU behavior)
        self.voice_cache = OrderedDict()
        self.max_cache_size = max_cache_size
        
        # Cache for generated audio
        self.audio_cache = {}
    
    def get_voice_state(self, voice):
        """Get voice state with caching."""
        # Move to end (most recently used)
        if voice in self.voice_cache:
            self.voice_cache.move_to_end(voice)
            print(f"Using cached voice state: {voice}")
        else:
            # If cache full, remove oldest
            if len(self.voice_cache) >= self.max_cache_size:
                oldest_voice = next(iter(self.voice_cache))
                del self.voice_cache[oldest_voice]
                print(f"Evicted from cache: {oldest_voice}")
            
            # Create new voice state
            voice_state = self.model.get_state_for_audio_prompt(voice)
            self.voice_cache[voice] = voice_state
            print(f"Created and cached voice state: {voice}")
        
        return self.voice_cache[voice]
    
    async def generate_audio_async(self, voice, text):
        """Asynchronous audio generation."""
        voice_state = self.get_voice_state(voice)
        
        # Check audio cache first
        cache_key = f"{voice}_{hash(text)}"
        if cache_key in self.audio_cache:
            print(f"Using cached audio: {cache_key}")
            return self.audio_cache[cache_key]
        
        # Generate audio
        start_time = time.time()
        audio = self.model.generate_audio(voice_state, text)
        generation_time = time.time() - start_time
        
        print(f"Generated {len(text)} chars in {generation_time:.2f}s")
        
        # Cache the result
        if len(self.audio_cache) >= 50:  # Limit audio cache size
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self.audio_cache))
            del self.audio_cache[oldest_key]
        
        self.audio_cache[cache_key] = audio
        
        return audio
    
    def clear_cache(self):
        """Clear all caches."""
        self.voice_cache.clear()
        self.audio_cache.clear()
        print("All caches cleared")

# Usage
tts = RealTimeTTS()

# Synchronous usage
audio1 = tts.generate_audio_async("alba", "First synchronous generation")
audio2 = tts.generate_audio_async("marius", "Second generation")

# Asynchronous usage
async def async_example():
    tasks = []
    texts = ["Hello from async 1", "Hello from async 2", "Hello from async 3"]
    
    for text in texts:
        task = asyncio.create_task(tts.generate_audio_async("alba", text))
        tasks.append(task)
    
    audios = await asyncio.gather(*tasks)
    print(f"Generated {len(audios)} audio files asynchronously")

# Run async example
asyncio.run(async_example())
```

### Streaming Audio Processing

```python
from pocket_tts import TTSModel
import sounddevice as sd
import numpy as np
import threading
import queue

class StreamingTTS:
    def __init__(self):
        self.model = TTSModel.load_model()
        self.model.compile_for_inference()
        self.voice_state = None
        self.audio_queue = queue.Queue()
        self.playing = False
    
    def set_voice(self, voice):
        """Set voice for streaming."""
        self.voice_state = self.model.get_state_for_audio_prompt(voice)
        print(f"Voice set to: {voice}")
    
    def start_streaming(self):
        """Start background streaming thread."""
        def stream_worker():
            while True:
                try:
                    text = input("Enter text (or 'quit' to exit): ")
                    if text.lower() == 'quit':
                        break
                    
                    if not text.strip():
                        continue
                    
                    print(f"Generating: {text}")
                    
                    # Stream generation
                    audio_chunks = []
                    for chunk in self.model.generate_audio_stream(self.voice_state, text):
                        audio_chunks.append(chunk)
                        # Queue chunk for immediate playback
                        self.audio_queue.put(chunk.numpy())
                    
                    # Combine and play
                    if audio_chunks:
                        full_audio = np.concatenate([chunk.numpy() for chunk in audio_chunks])
                        self.audio_queue.put(full_audio)
                    
                except Exception as e:
                    print(f"Error: {e}")
        
        # Start streaming thread
        stream_thread = threading.Thread(target=stream_worker, daemon=True)
        stream_thread.start()
        
        # Start playback thread
        playback_thread = threading.Thread(target=self.playback_worker, daemon=True)
        playback_thread.start()
        
        try:
            stream_thread.join()
            playback_thread.join()
        except KeyboardInterrupt:
            print("\nStopping streaming...")
    
    def playback_worker(self):
        """Background thread for audio playback."""
        while True:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                
                if len(audio_data.shape) > 1:
                    # If multi-dimensional, take first channel
                    audio_data = audio_data[0]
                
                sd.play(audio_data, self.model.sample_rate)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Playback error: {e}")

# Usage
streaming_tts = StreamingTTS()
streaming_tts.set_voice("alba")
streaming_tts.start_streaming()
```

For more advanced usage, see [Python API documentation](python-api.md) for direct integration with TTS model, [Configuration Guide](configuration.md) for optimization options, and [Integration Examples](integration-examples.md) for complete application examples.
