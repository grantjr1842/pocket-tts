# Pocket TTS Performance Optimization Guide

This guide covers advanced techniques for optimizing Pocket TTS performance, including model compilation, memory management, CPU optimization, and system resource monitoring.

## Table of Contents

- [Overview](#overview)
- [Model Compilation](#model-compilation)
- [Memory Management](#memory-management)
- [CPU Optimization](#cpu-optimization)
- [Generation Speed Optimization](#generation-speed-optimization)
- [Batch Processing](#batch-processing)
- [System Resource Monitoring](#system-resource-monitoring)
- [Performance Profiling](#performance-profiling)
- [Best Practices](#best-practices)

## Overview

Pocket TTS is designed to run efficiently on CPUs, but there are many techniques you can use to optimize performance further:

- **Compilation**: Use `torch.compile()` for 1.3-1.5x speedup
- **Memory Management**: Stream long texts, cache voice states, truncate audio prompts
- **CPU Optimization**: Limit threads, use affinity, optimize thread count
- **Batch Processing**: Reuse model and voice states across multiple generations
- **Monitoring**: Profile and monitor resource usage to identify bottlenecks

## Model Compilation

### Basic Compilation

Pocket TTS supports PyTorch compilation for improved inference speed:

```python
from pocket_tts import TTSModel

model = TTSModel.load_model()

# Basic compilation (recommended)
model.compile_for_inference()

# Now generate audio - it will be faster
voice_state = model.get_state_for_audio_prompt("alba")
audio = model.generate_audio(voice_state, "Hello world!")
```

### Compilation Modes

Different compilation modes offer trade-offs between compilation time and runtime speed:

```python
# Reduce overhead mode (fastest compilation, good speedup)
model.compile_for_inference(mode="reduce-overhead")

# Default mode (slower compilation, better speedup)
model.compile_for_inference(mode="default")

# Max autotune mode (slowest compilation, best speedup)
model.compile_for_inference(mode="max-autotune")
```

**Performance Comparison:**

| Mode | Compilation Time | Runtime Speedup | Best For |
|------|------------------|-----------------|----------|
| None | 0s | 1.0x (baseline) | Quick testing |
| reduce-overhead | ~10s | 1.3x | Production servers |
| default | ~20s | 1.4x | High-throughput |
| max-autotune | ~60s | 1.5x | Maximum performance |

### Advanced Compilation Options

```python
model.compile_for_inference(
    backend="inductor",          # Compilation backend
    mode="reduce-overhead",      # Compilation mode
    fullgraph=False,             # Require full graph capture
    dynamic=False,               # Dynamic shapes
    targets="all"                # Compile targets
)
```

### Selective Compilation

Compile specific components for targeted optimization:

```python
# Compile only flow-lm (language model)
model.compile_for_inference(targets="flow-lm")

# Compile only mimi-decoder
model.compile_for_inference(targets="mimi-decoder")

# Compile all components (default)
model.compile_for_inference(targets="all")
```

### When to Use Compilation

**Use compilation when:**
- Running a production server with many requests
- Processing many generations in a batch
- Need consistent, predictable performance
- Can afford initial compilation time

**Skip compilation when:**
- Doing quick testing or experimentation
- Only generating a few audio files
- Need to start up quickly
- Running on resource-constrained systems

### CLI Compilation

```bash
# Generate with compilation
pocket-tts generate --compile

# Server with compilation
pocket-tts serve --compile

# Custom compilation settings
pocket-tts generate \
  --compile \
  --compile-mode "max-autotune" \
  --compile-fullgraph
```

## Memory Management

### Streaming for Long Texts

For long texts, use streaming to reduce memory usage:

```python
from pocket_tts import TTSModel
import scipy.io.wavfile

model = TTSModel.load_model()
voice_state = model.get_state_for_audio_prompt("alba")

long_text = "..."  # Very long text

# Instead of generate_audio(), use generate_audio_stream()
all_chunks = []
for chunk in model.generate_audio_stream(voice_state, long_text):
    # Process each chunk immediately
    all_chunks.append(chunk)

# Combine if needed
import torch
full_audio = torch.cat(all_chunks, dim=0)
scipy.io.wavfile.write("output.wav", model.sample_rate, full_audio.numpy())
```

**Benefits:**
- Reduces peak memory usage by 50-70%
- Enables processing of arbitrarily long texts
- Allows real-time playback during generation

### Voice State Truncation

Truncate long audio prompts to save memory:

```python
# Full audio prompt (uses more memory)
voice_state = model.get_state_for_audio_prompt(
    "long_audio.wav",
    truncate=False
)

# Truncated to ~30 seconds (recommended for long audio)
voice_state = model.get_state_for_audio_prompt(
    "long_audio.wav",
    truncate=True
)
```

### Voice State Caching

Cache frequently used voice states to avoid recomputation:

```python
from functools import lru_cache

class VoiceCache:
    def __init__(self, model):
        self.model = model
        self._cache = {}

    def get_voice(self, voice_name):
        """Get cached voice state."""
        if voice_name not in self._cache:
            print(f"Loading voice: {voice_name}")
            self._cache[voice_name] = self.model.get_state_for_audio_prompt(voice_name)
        else:
            print(f"Using cached voice: {voice_name}")
        return self._cache[voice_name]

    def clear(self):
        """Clear cache to free memory."""
        self._cache.clear()

# Usage
model = TTSModel.load_model()
voice_cache = VoiceCache(model)

# First call loads the voice
voice1 = voice_cache.get_voice("alba")

# Subsequent calls use cache
voice2 = voice_cache.get_voice("alba")

# Clear cache when done
voice_cache.clear()
```

### Memory Cleanup

Explicitly clean up resources after operations:

```python
import gc
import torch

def generate_with_cleanup(text: str):
    """Generate audio and clean up resources."""
    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt("alba")

    try:
        audio = model.generate_audio(voice_state, text)
        return audio
    finally:
        # Clean up
        del voice_state
        del audio
        gc.collect()
        torch.cuda.empty_cache()  # If using CUDA

# Usage
audio = generate_with_cleanup("Hello world!")
```

### Batch Processing with Memory Management

Process multiple texts with memory-conscious batching:

```python
def batch_generate(texts, batch_size=5):
    """Generate multiple texts with memory management."""
    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt("alba")

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        # Process batch
        for text in batch:
            audio = model.generate_audio(voice_state, text)
            results.append(audio)

        # Clean up between batches
        gc.collect()

    return results

# Usage
texts = ["Text 1", "Text 2", "Text 3", ...]
results = batch_generate(texts, batch_size=5)
```

## CPU Optimization

### Thread Count Optimization

Limit PyTorch threads for optimal performance:

```python
import os
import torch

# Set thread count (recommended: 2-4 for best performance)
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
torch.set_num_threads(4)

from pocket_tts import TTSModel
model = TTSModel.load_model()
```

**Guidelines:**
- **1-2 cores**: Low-end systems, background tasks
- **2-4 cores**: Recommended for most systems
- **4-8 cores**: High-end systems, dedicated workstations
- **8+ cores**: Diminishing returns, may cause contention

### CPU Affinity

Bind process to specific CPU cores for consistent performance:

```bash
# Linux: Use taskset to bind to specific cores
taskset -c 0-3 pocket-tts serve

# Bind to specific cores
taskset -c 4,5,6,7 pocket-tts generate --text "Hello"
```

### Process Priority

Adjust process priority to manage resource contention:

```bash
# Linux: Run with lower priority (nice)
nice -n 10 pocket-tts serve

# Linux: Run with higher priority
nice -n -5 pocket-tts serve

# Windows: Use start with priority
start /low pocket-tts serve
start /high pocket-tts serve
```

### Disable Unwanted Features

Disable AVX/AVX2 if causing issues:

```python
import os
# Disable AVX2
os.environ['TORCH_DISABLE_AVX2'] = '1'

from pocket_tts import TTSModel
model = TTSModel.load_model()
```

## Generation Speed Optimization

### Parameter Tuning

Adjust generation parameters for speed vs quality trade-off:

```python
# Fastest (lower quality)
model = TTSModel.load_model(
    temp=0.7,
    lsd_decode_steps=1,  # Single step
    noise_clamp=1.0      # Clamp for speed
)

# Balanced (recommended)
model = TTSModel.load_model(
    temp=0.7,
    lsd_decode_steps=4,
    noise_clamp=None
)

# High quality (slower)
model = TTSModel.load_model(
    temp=0.8,
    lsd_decode_steps=8,
    noise_clamp=None
)
```

**Performance Impact:**
- 1 step: ~6x real-time (fastest)
- 4 steps: ~1.5x real-time (recommended)
- 8 steps: ~0.8x real-time (slower)
- 16 steps: ~0.4x real-time (slowest)

### Combined Optimization

Combine multiple techniques for maximum speed:

```python
import os
import torch

# Optimize threads
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
torch.set_num_threads(4)

from pocket_tts import TTSModel

# Load model with fast parameters
model = TTSModel.load_model(
    temp=0.7,
    lsd_decode_steps=4
)

# Compile for inference
model.compile_for_inference(mode="reduce-overhead")

# Cache voice state
voice_state = model.get_state_for_audio_prompt("alba")

# Generate
audio = model.generate_audio(voice_state, "Hello world!")
```

## Batch Processing

### Efficient Batch Processing

Reuse model and voice states across multiple generations:

```python
def batch_generate_optimized(texts, voice="alba"):
    """Efficiently generate multiple texts."""
    # Load model once
    model = TTSModel.load_model()
    model.compile_for_inference(mode="reduce-overhead")

    # Load voice state once
    voice_state = model.get_state_for_audio_prompt(voice)

    results = []
    for text in texts:
        audio = model.generate_audio(voice_state, text)
        results.append(audio)

    return results

# Usage
texts = ["Hello", "World", "How are you?"]
results = batch_generate_optimized(texts)
```

### Parallel Batch Processing (Multiple Models)

Use multiple model instances for parallel processing:

```python
from concurrent.futures import ThreadPoolExecutor
import threading

class ThreadSafeTTS:
    """Thread-safe TTS wrapper."""
    def __init__(self):
        self.model = TTSModel.load_model()
        self.model.compile_for_inference()
        self.lock = threading.Lock()

    def generate(self, voice_state, text):
        """Generate audio with thread safety."""
        with self.lock:
            return self.model.generate_audio(voice_state, text)

def parallel_generate(texts, voice="alba", max_workers=2):
    """Generate texts in parallel using multiple TTS instances."""
    def create_instance():
        tts = ThreadSafeTTS()
        voice_state = tts.model.get_state_for_audio_prompt(voice)
        return tts, voice_state

    # Create instances
    instances = [create_instance() for _ in range(max_workers)]

    results = [None] * len(texts)

    def worker(i, text, instance, voice_state):
        results[i] = instance.generate(voice_state, text)

    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, text in enumerate(texts):
            instance, voice_state = instances[i % max_workers]
            future = executor.submit(worker, i, text, instance, voice_state)
            futures.append(future)

        # Wait for completion
        for future in futures:
            future.result()

    return results

# Usage
texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
results = parallel_generate(texts, max_workers=2)
```

## System Resource Monitoring

### Memory Monitoring

Monitor memory usage during generation:

```python
import psutil
import gc

def monitor_memory():
    """Monitor memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size
        "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size
        "percent": process.memory_percent()
    }

# Usage
print("Before:", monitor_memory())
model = TTSModel.load_model()
print("After load:", monitor_memory())

voice_state = model.get_state_for_audio_prompt("alba")
print("After voice:", monitor_memory())

audio = model.generate_audio(voice_state, "Hello world!")
print("After generate:", monitor_memory())

# Cleanup
del audio
gc.collect()
print("After cleanup:", monitor_memory())
```

### CPU Monitoring

Monitor CPU usage during operations:

```python
import psutil
import time

def monitor_cpu(duration=5):
    """Monitor CPU usage over time."""
    cpu_percent = []
    start_time = time.time()

    while time.time() - start_time < duration:
        cpu = psutil.cpu_percent(interval=0.1)
        cpu_percent.append(cpu)

    return {
        "avg": sum(cpu_percent) / len(cpu_percent),
        "max": max(cpu_percent),
        "min": min(cpu_percent)
    }

# Usage
print("Starting generation...")
cpu_stats = monitor_cpu(duration=3)
print(f"CPU: Avg={cpu_stats['avg']:.1f}%, Max={cpu_stats['max']:.1f}%")
```

### Generation Speed Tracking

Track real-time factor (RTF) - ratio of generation time to audio duration:

```python
import time

def measure_rtf(model, voice_state, text):
    """Measure real-time factor."""
    start_time = time.time()
    audio = model.generate_audio(voice_state, text)
    generation_time = time.time() - start_time

    audio_duration = audio.shape[0] / model.sample_rate
    rtf = generation_time / audio_duration

    return {
        "generation_time": generation_time,
        "audio_duration": audio_duration,
        "rtf": rtf,
        "speed_ratio": 1 / rtf if rtf > 0 else 0
    }

# Usage
stats = measure_rtf(model, voice_state, "Hello world!")
print(f"Generation time: {stats['generation_time']:.2f}s")
print(f"Audio duration: {stats['audio_duration']:.2f}s")
print(f"RTF: {stats['rtf']:.2f}")
print(f"Speed: {stats['speed_ratio']:.1f}x real-time")
```

### Comprehensive Monitoring

Create a comprehensive monitoring system:

```python
class TTSPerformanceMonitor:
    """Monitor TTS performance metrics."""

    def __init__(self):
        self.metrics = []

    def track_generation(self, model, voice_state, text, metadata=None):
        """Track a generation with performance metrics."""
        import psutil
        import time

        # Record initial state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)

        # Time generation
        start_time = time.time()
        audio = model.generate_audio(voice_state, text)
        generation_time = time.time() - start_time

        # Record final state
        final_memory = process.memory_info().rss / (1024 * 1024)

        # Calculate metrics
        audio_duration = audio.shape[0] / model.sample_rate
        rtf = generation_time / audio_duration

        metric = {
            "text_length": len(text),
            "audio_samples": audio.shape[0],
            "audio_duration": audio_duration,
            "generation_time": generation_time,
            "rtf": rtf,
            "speed_ratio": 1 / rtf if rtf > 0 else 0,
            "memory_mb": final_memory - initial_memory,
            "metadata": metadata or {}
        }

        self.metrics.append(metric)
        return metric

    def get_summary(self):
        """Get summary of all tracked generations."""
        if not self.metrics:
            return {}

        return {
            "count": len(self.metrics),
            "avg_rtf": sum(m["rtf"] for m in self.metrics) / len(self.metrics),
            "avg_speed": sum(m["speed_ratio"] for m in self.metrics) / len(self.metrics),
            "avg_generation_time": sum(m["generation_time"] for m in self.metrics) / len(self.metrics),
            "avg_memory_mb": sum(m["memory_mb"] for m in self.metrics) / len(self.metrics),
        }

    def print_summary(self):
        """Print performance summary."""
        summary = self.get_summary()
        print("\n=== Performance Summary ===")
        print(f"Total generations: {summary['count']}")
        print(f"Average RTF: {summary['avg_rtf']:.3f}")
        print(f"Average speed: {summary['avg_speed']:.2f}x real-time")
        print(f"Average generation time: {summary['avg_generation_time']:.2f}s")
        print(f"Average memory usage: {summary['avg_memory_mb']:.1f}MB")

# Usage
monitor = TTSPerformanceMonitor()

model = TTSModel.load_model()
voice_state = model.get_state_for_audio_prompt("alba")

texts = ["Hello", "World", "How are you?"]
for text in texts:
    monitor.track_generation(model, voice_state, text)

monitor.print_summary()
```

## Performance Profiling

### Profile Generation Performance

Use Python's profiler to identify bottlenecks:

```python
import cProfile
import pstats
from io import StringIO

def profile_generation(model, voice_state, text):
    """Profile TTS generation."""
    pr = cProfile.Profile()
    pr.enable()

    # Generate audio
    audio = model.generate_audio(voice_state, text)

    pr.disable()

    # Print statistics
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print(s.getvalue())

    return audio

# Usage
model = TTSModel.load_model()
voice_state = model.get_state_for_audio_prompt("alba")
audio = profile_generation(model, voice_state, "Hello world!")
```

### Line Profiling

Profile line-by-line for detailed analysis:

```bash
# Install line_profiler
pip install line_profiler

# Add @profile decorator to functions you want to profile
# Then run:
kernprof -l -v your_script.py
```

## Best Practices

### 1. Always Compile in Production

```python
model = TTSModel.load_model()
model.compile_for_inference(mode="reduce-overhead")
```

### 2. Cache Voice States

```python
# Load voice states once
voices = {
    "alba": model.get_state_for_audio_prompt("alba"),
    "marius": model.get_state_for_audio_prompt("marius"),
}

# Reuse across generations
for text in texts:
    audio = model.generate_audio(voices["alba"], text)
```

### 3. Use Streaming for Long Texts

```python
# For long texts (>500 characters)
for chunk in model.generate_audio_stream(voice_state, long_text):
    # Process chunk immediately
    pass
```

### 4. Limit Thread Count

```python
import os
import torch

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
torch.set_num_threads(4)
```

### 5. Monitor Performance

```python
# Track RTF for each generation
stats = measure_rtf(model, voice_state, text)
if stats['rtf'] > 0.5:  # Slower than 2x real-time
    print("Warning: Generation slower than expected")
```

### 6. Clean Up Resources

```python
# Always clean up after operations
try:
    audio = model.generate_audio(voice_state, text)
finally:
    del audio
    gc.collect()
```

### 7. Use Appropriate Quality Settings

```python
# Choose settings based on use case
if speed_critical:
    model = TTSModel.load_model(lsd_decode_steps=1)
elif quality_critical:
    model = TTSModel.load_model(lsd_decode_steps=8)
else:
    model = TTSModel.load_model(lsd_decode_steps=4)  # Balanced
```

For more information, see:
- [Configuration Guide](configuration-guide.md)
- [Python API Documentation](python-api.md)
- [Troubleshooting Guide](troubleshooting.md)
