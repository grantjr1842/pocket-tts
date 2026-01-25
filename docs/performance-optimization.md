# Performance Optimization Guide

This guide covers performance optimization techniques for Pocket TTS, including memory management, compilation, and deployment strategies.

## Performance Fundamentals

### Understanding Bottlenecks

Pocket TTS performance is typically limited by:

1. **CPU-bound operations**: Model inference (primary bottleneck)
2. **Memory allocation**: Audio tensor creation and copying
3. **File I/O**: Voice loading and audio saving
4. **Model loading**: Cold start initialization time

### Performance Targets

| Operation | Target | Typical Performance |
|-----------|---------|-------------------|
| Model loading | < 3 seconds | 1-2 seconds on modern CPU |
| Voice preparation | < 1 second | 200-500ms |
| Audio generation | 6x real-time | 1 second audio in ~170ms |
| First audio chunk | < 200ms | Streaming latency |

## Model Compilation

### Torch Compilation Basics

```python
from pocket_tts import TTSModel
import time

# Benchmark without compilation
model = TTSModel.load_model()
voice_state = model.get_state_for_audio_prompt("alba")

start_time = time.time()
audio = model.generate_audio(voice_state, "Performance test text.")
baseline_time = time.time() - start_time
print(f"Baseline: {baseline_time:.3f}s")

# Enable compilation
model.compile_for_inference()

# Benchmark with compilation
start_time = time.time()
audio_compiled = model.generate_audio(voice_state, "Performance test text.")
compiled_time = time.time() - start_time
print(f"Compiled: {compiled_time:.3f}s")
print(f"Speedup: {baseline_time/compiled_time:.2f}x")
```

### Advanced Compilation Strategies

```python
from pocket_tts import TTSModel

model = TTSModel.load_model()

# Strategy 1: Maximum performance
model.compile_for_inference(
    backend="inductor",
    mode="reduce-overhead",
    fullgraph=True,          # Capture entire computation graph
    dynamic=False,           # Static shapes for optimization
    targets=["flow-lm", "mimi-decoder"]
)

# Strategy 2: Memory optimization
model.compile_for_inference(
    backend="inductor",
    mode="max-autotune",      # Find optimal kernels
    fullgraph=False,          # Chunked processing
    dynamic=False,
    targets=["flow-lm"]       # Compile only bottleneck module
)

# Strategy 3: Development friendly
model.compile_for_inference(
    backend="aot_eager",      # Faster compilation
    mode="default",
    fullgraph=False,
    dynamic=True,            # Flexible input sizes
    targets="all"
)
```

### Compilation Warm-up

```python
def warmup_model(model, voice_state):
    """Warm up compiled model for optimal performance."""
    
    warmup_texts = [
        "Warm up text one.",
        "Second warm up sentence.",
        "Final warm up phrase."
    ]
    
    print("Warming up model...")
    with torch.no_grad():  # Disable gradient computation
        for i, text in enumerate(warmup_texts):
            _ = model.generate_audio(voice_state, text, copy_state=False)
            print(f"  Warmup {i+1}/3 complete")
    
    print("Model warmed up and ready!")

# Usage
model = TTSModel.load_model()
model.compile_for_inference()
voice_state = model.get_state_for_audio_prompt("alba")
warmup_model(model, voice_state)
```

## Memory Management

### Voice State Caching

```python
from functools import lru_cache
from pocket_tts import TTSModel

class OptimizedTTS:
    def __init__(self):
        self.model = TTSModel.load_model()
        self.model.compile_for_inference()
        self._voice_cache = {}
    
    def get_voice_state(self, voice):
        """Get cached voice state or create new one."""
        if voice not in self._voice_cache:
            print(f"Loading voice: {voice}")
            self._voice_cache[voice] = self.model.get_state_for_audio_prompt(voice)
        else:
            print(f"Using cached voice: {voice}")
        
        return self._voice_cache[voice]
    
    def generate_audio(self, voice, text):
        """Generate audio with cached voice state."""
        voice_state = self.get_voice_state(voice)
        return self.model.generate_audio(voice_state, text, copy_state=True)

# Usage
tts = OptimizedTTS()

# First call loads voice
audio1 = tts.generate_audio("alba", "First sentence")

# Second call uses cached voice
audio2 = tts.generate_audio("alba", "Second sentence")
```

### Memory Pool Management

```python
import torch
import numpy as np

class AudioPool:
    """Memory pool for audio tensors to reduce allocation overhead."""
    
    def __init__(self, pool_size=10, max_samples=240000):  # 10 seconds max
        self.pool_size = pool_size
        self.max_samples = max_samples
        self.available_tensors = []
        self.in_use_tensors = set()
        
        # Pre-allocate tensors
        for _ in range(pool_size):
            tensor = torch.zeros(max_samples, dtype=torch.float32)
            self.available_tensors.append(tensor)
    
    def acquire(self, num_samples):
        """Get tensor from pool or allocate new one."""
        if num_samples > self.max_samples:
            return torch.zeros(num_samples, dtype=torch.float32)
        
        for i, tensor in enumerate(self.available_tensors):
            if i not in self.in_use_tensors:
                # Trim to required size
                trimmed = tensor[:num_samples]
                self.in_use_tensors.add(i)
                return trimmed
        
        # Pool exhausted, allocate new tensor
        return torch.zeros(num_samples, dtype=torch.float32)
    
    def release(self, tensor):
        """Return tensor to pool."""
        # Find tensor in pool and mark as available
        tensor_size = tensor.shape[0]
        for i, pool_tensor in enumerate(self.available_tensors):
            if pool_tensor.shape[0] >= tensor_size:
                if i in self.in_use_tensors:
                    self.in_use_tensors.remove(i)
                    # Zero out for reuse
                    pool_tensor[:tensor_size] = 0
                    return

# Usage
audio_pool = AudioPool()

def optimized_generation(model, voice_state, text):
    audio = model.generate_audio(voice_state, text)
    # Process audio...
    
    # Return to pool if applicable
    if hasattr(audio, '_pooled'):
        audio_pool.release(audio)
```

## Batch Processing Optimization

### Efficient Voice Management

```python
class VoiceManager:
    """Efficiently manage multiple voices for batch processing."""
    
    def __init__(self, voices=None):
        self.model = TTSModel.load_model()
        self.model.compile_for_inference()
        self.voices = {}
        
        # Preload voices if specified
        if voices:
            self.preload_voices(voices)
    
    def preload_voices(self, voice_list):
        """Preload multiple voice states."""
        print(f"Preloading {len(voice_list)} voices...")
        for voice in voice_list:
            self.voices[voice] = self.model.get_state_for_audio_prompt(voice)
        print("Voice preloading complete.")
    
    def generate_batch(self, requests):
        """Generate audio for multiple requests efficiently."""
        results = []
        
        for voice, text in requests:
            # Reuse cached voice state
            voice_state = self.voices.get(voice)
            if not voice_state:
                voice_state = self.model.get_state_for_audio_prompt(voice)
                self.voices[voice] = voice_state  # Cache for future
            
            audio = self.model.generate_audio(voice_state, text, copy_state=True)
            results.append(audio)
        
        return results

# Usage
manager = VoiceManager(voices=["alba", "marius", "javert"])

requests = [
    ("alba", "Hello from Alba."),
    ("marius", "Marius speaking here."),
    ("javert", "This is Javert.")
]

audios = manager.generate_batch(requests)
```

### Streaming Optimization

```python
def optimized_streaming_generation(model, voice_state, text, chunk_size=50):
    """Optimized streaming with efficient chunk handling."""
    
    # Pre-allocate output buffer
    total_audio = []
    current_position = 0
    
    print(f"Starting streaming generation for {len(text)} characters...")
    
    for chunk in model.generate_audio_stream(voice_state, text):
        total_audio.append(chunk)
        current_position += chunk.shape[0]
        
        # Progress reporting
        progress = (current_position / (len(text) * 50)) * 100  # Estimate
        print(f"  Generated: {progress:.1f}% ({current_position} samples)")
        
        # Yield chunks for real-time processing if needed
        yield chunk
    
    final_audio = torch.cat(total_audio, dim=0)
    print(f"Streaming complete: {final_audio.shape[0]} samples total")
    return final_audio

# Usage
for chunk in optimized_streaming_generation(model, voice_state, long_text):
    # Process chunk in real-time
    process_audio_chunk(chunk)
```

## System-Level Optimization

### CPU Affinity and Threading

```python
import os
import psutil
import torch

def optimize_system_settings():
    """Optimize system settings for TTS performance."""
    
    # Set CPU affinity to avoid context switching
    cpu_count = os.cpu_count()
    if cpu_count >= 4:
        # Use dedicated cores for TTS
        os.sched_setaffinity(0, [0, 1])  # Use first 2 cores
    
    # Optimize PyTorch threading
    optimal_threads = min(2, cpu_count)  # Pocket TTS works best with 1-2 threads
    torch.set_num_threads(optimal_threads)
    
    # Set inter-op threads to 0 for automatic
    torch.set_num_interop_threads(0)
    
    print(f"System optimized: {optimal_threads} threads, CPU affinity set")

# Early in your application
optimize_system_settings()
model = TTSModel.load_model()
```

### Memory Optimization

```python
import gc
import torch

def optimize_memory_usage():
    """Configure memory settings for optimal performance."""
    
    # Enable memory-efficient settings
    torch.backends.cuda.matmul.allow_tf32 = False  # More precise but slower
    torch.backends.cudnn.allow_tf32 = False
    
    # Enable memory mapping for large tensors
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Configure garbage collection
    gc.set_threshold(700, 10, 10)  # More aggressive GC
    
    print("Memory optimization configured")

def cleanup_memory():
    """Explicit memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Usage pattern
optimize_memory_usage()

try:
    model = TTSModel.load_model()
    # Your generation code here
finally:
    cleanup_memory()
```

## Deployment Optimization

### Production Configuration

```python
class ProductionTTS:
    """Production-optimized TTS with monitoring and fallbacks."""
    
    def __init__(self):
        self.model = self._load_optimized_model()
        self.voice_cache = {}
        self.stats = {"generations": 0, "errors": 0, "total_time": 0}
    
    def _load_optimized_model(self):
        """Load model with production optimizations."""
        try:
            model = TTSModel.load_model()
            model.compile_for_inference(
                backend="inductor",
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=False
            )
            
            # Warm up model
            voice_state = model.get_state_for_audio_prompt("alba")
            for _ in range(3):  # Warmup iterations
                model.generate_audio(voice_state, "Warm up.", copy_state=False)
            
            return model
            
        except Exception as e:
            print(f"Optimized loading failed: {e}")
            # Fallback to basic configuration
            return TTSModel.load_model(temperature=0.7)
    
    @monitor_performance
    def generate(self, voice, text):
        """Generate audio with performance monitoring."""
        try:
            start_time = time.time()
            
            voice_state = self._get_cached_voice(voice)
            audio = self.model.generate_audio(voice_state, text, copy_state=True)
            
            duration = time.time() - start_time
            self.stats["generations"] += 1
            self.stats["total_time"] += duration
            
            return audio
            
        except Exception as e:
            self.stats["errors"] += 1
            raise
    
    def get_performance_stats(self):
        """Get current performance statistics."""
        gens = self.stats["generations"]
        if gens == 0:
            return {}
        
        return {
            "total_generations": gens,
            "total_errors": self.stats["errors"],
            "average_time": self.stats["total_time"] / gens,
            "success_rate": (gens - self.stats["errors"]) / gens * 100
        }
```

### Load Testing and Scaling

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

def load_test_tts(max_workers=4, requests_per_worker=10):
    """Load test TTS performance under concurrent load."""
    
    tts = ProductionTTS()
    
    def worker_test(worker_id):
        """Test worker that generates multiple audio samples."""
        results = []
        
        for i in range(requests_per_worker):
            try:
                start_time = time.time()
                audio = tts.generate("alba", f"Worker {worker_id}, request {i}")
                duration = time.time() - start_time
                results.append({"worker": worker_id, "request": i, "duration": duration, "success": True})
            except Exception as e:
                results.append({"worker": worker_id, "request": i, "error": str(e), "success": False})
        
        return results
    
    # Run concurrent workers
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_test, i) for i in range(max_workers)]
        all_results = []
        for future in futures:
            all_results.extend(future.result())
    
    total_time = time.time() - start_time
    successful = sum(1 for r in all_results if r["success"])
    total_requests = len(all_results)
    
    print(f"Load test complete:")
    print(f"  Total requests: {total_requests}")
    print(f"  Successful: {successful} ({successful/total_requests*100:.1f}%)")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Requests/second: {total_requests/total_time:.2f}")
    print(f"  Average latency: {total_time/total_requests:.3f}s")
    
    return all_results

# Usage
if __name__ == "__main__":
    load_test_results = load_test_tts(max_workers=2, requests_per_worker=5)
```

## Performance Monitoring

### Real-time Monitoring

```python
import psutil
import threading
import time

class PerformanceMonitor:
    """Monitor system performance during TTS operations."""
    
    def __init__(self):
        self.monitoring = False
        self.stats = {
            "cpu_usage": [],
            "memory_usage": [],
            "generation_times": []
        }
    
    def start_monitoring(self):
        """Start background monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and return statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        return self._calculate_stats()
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            self.stats["cpu_usage"].append(psutil.cpu_percent())
            self.stats["memory_usage"].append(psutil.virtual_memory().percent)
            time.sleep(0.5)  # Sample every 500ms
    
    def record_generation(self, duration):
        """Record a generation time."""
        self.stats["generation_times"].append(duration)
    
    def _calculate_stats(self):
        """Calculate performance statistics."""
        if not self.stats["cpu_usage"]:
            return {}
        
        return {
            "avg_cpu": sum(self.stats["cpu_usage"]) / len(self.stats["cpu_usage"]),
            "max_cpu": max(self.stats["cpu_usage"]),
            "avg_memory": sum(self.stats["memory_usage"]) / len(self.stats["memory_usage"]),
            "max_memory": max(self.stats["memory_usage"]),
            "avg_generation_time": sum(self.stats["generation_times"]) / len(self.stats["generation_times"]) if self.stats["generation_times"] else 0,
            "total_generations": len(self.stats["generation_times"])
        }

# Usage
monitor = PerformanceMonitor()
monitor.start_monitoring()

# Perform TTS operations
start_time = time.time()
audio = model.generate_audio(voice_state, "Monitoring test text")
duration = time.time() - start_time

monitor.record_generation(duration)
stats = monitor.stop_monitoring()

print("Performance Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")
```

This performance optimization guide helps you get the most out of Pocket TTS in various deployment scenarios.