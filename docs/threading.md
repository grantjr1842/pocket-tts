# PyTorch Threading Configuration

## Overview

Pocket TTS uses PyTorch for text-to-speech generation. The number of CPU threads that PyTorch uses can significantly affect performance.

## Default Configuration

By default, Pocket TTS uses **1 thread** for PyTorch operations. This conservative default ensures:
- Predictable performance across different systems
- Avoidance of thread contention issues
- Consistent behavior for single-threaded workloads

## Configuring Thread Count

### Environment Variable

Set the `POCKET_TTS_NUM_THREADS` environment variable to change the thread count:

```bash
# Use 2 threads
export POCKET_TTS_NUM_THREADS=2
pocket-tts generate "Hello world"

# Use 4 threads
export POCKET_TTS_NUM_THREADS=4
pocket-tts generate "Hello world"
```

### In Python Code

```python
import os
os.environ["POCKET_TTS_NUM_THREADS"] = "2"

from pocket_tts import TTSModel
model = TTSModel.load_model()
```

## Performance Considerations

### When to Use More Threads

**Multiple threads may help when:**
- Running on multi-core systems (4+ cores)
- Generating long texts
- System is not CPU-constrained
- Other processes are not competing for CPU

**Single thread is better when:**
- Running on systems with 1-2 cores
- System is under heavy CPU load
- Generating short texts (threading overhead may outweigh benefits)
- Running multiple TTS instances in parallel

### Benchmarking

Use the provided benchmark script to test different thread counts on your system:

```bash
python scripts/benchmark_threads.py
```

This will test 1, 2, and 4 threads with texts of different lengths and show:
- Generation time for each configuration
- Real-time factor (how fast generation is compared to playback speed)

### Example Benchmark Results

Results vary significantly by hardware and workload:

| Threads | Short Text | Medium Text | Long Text |
|---------|------------|-------------|-----------|
| 1       | 0.50s      | 2.0s        | 8.0s      |
| 2       | 0.45s      | 1.6s        | 6.0s      |
| 4       | 0.48s      | 1.7s        | 5.8s      |

**Note:** These are example results only. Run the benchmark on your system for accurate data.

## Thread-Safe Usage

When running multiple TTS instances in parallel (e.g., in a web server), consider:

1. **Limiting total threads**: If running 4 instances with 4 threads each, that's 16 threads total
2. **Using single thread per instance**: Often more efficient for concurrent workloads
3. **Monitoring CPU usage**: Ensure system isn't oversubscribed

Example for concurrent usage:

```bash
# For a server handling 4 concurrent requests
export POCKET_TTS_NUM_THREADS=1  # Each instance uses 1 thread
# Total: 4 threads = efficient resource usage
```

## Implementation Details

- Thread count is set when the `pocket_tts.models.tts_model` module is first imported
- Changing `POCKET_TTS_NUM_THREADS` requires restarting the Python process
- The setting affects all PyTorch operations (not just TTS generation)
- Thread count is global per Python process, not per model instance

## See Also

- [PyTorch documentation on threading](https://pytorch.org/docs/stable/cpu_threading_torchscript_inference.html)
- [scripts/benchmark_threads.py](scripts/benchmark_threads.py) - Benchmarking script
- [comprehensive-optimization-plan.md](comprehensive-optimization-plan.md) - Performance optimization roadmap
