# Pocket TTS Error Handling Guide

This guide provides comprehensive error handling patterns and best practices for building robust applications with Pocket TTS.

## Table of Contents

- [Overview](#overview)
- [Common Exceptions](#common-exceptions)
- [Error Handling Patterns](#error-handling-patterns)
- [Retry Strategies](#retry-strategies)
- [Validation Techniques](#validation-techniques)
- [Graceful Degradation](#graceful-degradation)
- [Logging and Monitoring](#logging-and-monitoring)
- [Complete Examples](#complete-examples)

## Overview

Pocket TTS can raise various exceptions during model loading, voice cloning, and audio generation. Proper error handling ensures your application remains stable and provides good user experience.

### Key Principles

1. **Fail Fast**: Validate inputs early
2. **Graceful Degradation**: Provide fallbacks when possible
3. **Informative Messages**: Help users understand and fix issues
4. **Resource Cleanup**: Always release resources in finally blocks
5. **Logging**: Record errors for debugging

## Common Exceptions

### Model Loading Errors

```python
from pocket_tts import TTSModel

try:
    model = TTSModel.load_model()
except FileNotFoundError as e:
    # Model files not found
    print(f"Model file not found: {e}")
except RuntimeError as e:
    # Download or initialization failed
    print(f"Failed to initialize model: {e}")
except Exception as e:
    # Other unexpected errors
    print(f"Unexpected error loading model: {type(e).__name__}: {e}")
```

### Voice Cloning Errors

```python
try:
    voice_state = model.get_state_for_audio_prompt("invalid_path.wav")
except FileNotFoundError:
    print("Audio file not found")
except ValueError as e:
    # Invalid audio format or corrupted file
    print(f"Invalid audio file: {e}")
except RuntimeError as e:
    # Audio processing failed
    print(f"Failed to process audio: {e}")
```

### Audio Generation Errors

```python
try:
    audio = model.generate_audio(voice_state, "Hello world")
except ValueError as e:
    # Invalid input text or voice state
    print(f"Invalid input: {e}")
except RuntimeError as e:
    # Generation failed (out of memory, etc.)
    print(f"Generation failed: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
```

### Network Errors

```python
try:
    voice_state = model.get_state_for_audio_prompt(
        "https://example.com/voice.wav"
    )
except requests.exceptions.RequestException as e:
    # Network-related errors
    print(f"Network error: {e}")
except Exception as e:
    print(f"Failed to download voice: {e}")
```

## Error Handling Patterns

### Pattern 1: Basic Try-Except

```python
from pocket_tts import TTSModel

def generate_speech(text, voice="alba"):
    """Generate speech with basic error handling."""
    try:
        model = TTSModel.load_model()
        voice_state = model.get_state_for_audio_prompt(voice)
        audio = model.generate_audio(voice_state, text)
        return audio
    except FileNotFoundError:
        print("Error: Model or voice file not found")
        return None
    except ValueError as e:
        print(f"Error: Invalid input - {e}")
        return None
    except Exception as e:
        print(f"Error: {type(e).__name__} - {e}")
        return None

# Usage
audio = generate_speech("Hello world")
if audio is not None:
    print("Generation successful!")
```

### Pattern 2: Context Manager

```python
from contextlib import contextmanager

@contextmanager
def tts_model(compile=False):
    """Context manager for TTS model with proper cleanup."""
    model = None
    try:
        model = TTSModel.load_model()
        if compile:
            model.compile_for_inference()
        yield model
    except Exception as e:
        print(f"Error in TTS operation: {e}")
        raise
    finally:
        # Cleanup if needed
        if model is not None:
            del model
        import gc
        gc.collect()

# Usage
try:
    with tts_model(compile=True) as model:
        voice_state = model.get_state_for_audio_prompt("alba")
        audio = model.generate_audio(voice_state, "Hello!")
except Exception as e:
    print(f"Failed to generate audio: {e}")
```

### Pattern 3: Result Type

```python
from typing import Union, Tuple
from dataclasses import dataclass

@dataclass
class Success:
    """Successful operation result."""
    value: any

@dataclass
class Error:
    """Error result."""
    message: str
    exception: Exception = None

Result = Union[Success, Error]

def safe_generate_audio(model, voice_state, text) -> Result:
    """Generate audio with result type."""
    try:
        audio = model.generate_audio(voice_state, text)
        return Success(audio)
    except FileNotFoundError:
        return Error("Model or voice file not found")
    except ValueError as e:
        return Error(f"Invalid input: {e}")
    except RuntimeError as e:
        return Error(f"Generation failed: {e}")
    except Exception as e:
        return Error(f"Unexpected error: {e}", e)

# Usage
model = TTSModel.load_model()
voice_state = model.get_state_for_audio_prompt("alba")
result = safe_generate_audio(model, voice_state, "Hello!")

if isinstance(result, Success):
    print(f"Generated {result.value.shape[0]} samples")
elif isinstance(result, Error):
    print(f"Error: {result.message}")
```

### Pattern 4: Custom Exception Hierarchy

```python
class TTSError(Exception):
    """Base exception for TTS errors."""
    pass

class ModelLoadError(TTSError):
    """Model loading failed."""
    pass

class VoiceLoadError(TTSError):
    """Voice loading failed."""
    pass

class GenerationError(TTSError):
    """Audio generation failed."""
    pass

class ValidationError(TTSError):
    """Input validation failed."""
    pass

def generate_speech_safe(text: str, voice: str = "alba") -> bytes:
    """Generate speech with custom exceptions."""
    # Validate inputs
    if not text or not text.strip():
        raise ValidationError("Text cannot be empty")
    if len(text) > 10000:
        raise ValidationError("Text too long (max 10000 characters)")

    try:
        model = TTSModel.load_model()
    except Exception as e:
        raise ModelLoadError(f"Failed to load model: {e}") from e

    try:
        voice_state = model.get_state_for_audio_prompt(voice)
    except Exception as e:
        raise VoiceLoadError(f"Failed to load voice '{voice}': {e}") from e

    try:
        audio = model.generate_audio(voice_state, text)
        return audio
    except Exception as e:
        raise GenerationError(f"Failed to generate audio: {e}") from e

# Usage
try:
    audio = generate_speech_safe("Hello world!", "alba")
except ValidationError as e:
    print(f"Validation error: {e}")
except ModelLoadError as e:
    print(f"Model error: {e}")
except VoiceLoadError as e:
    print(f"Voice error: {e}")
except GenerationError as e:
    print(f"Generation error: {e}")
```

### Pattern 5: Retry with Exponential Backoff

```python
import time
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry(max_attempts=3, initial_delay=1, backoff_factor=2):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"All {max_attempts} attempts failed")

            raise last_exception

        return wrapper
    return decorator

@retry(max_attempts=3, initial_delay=1, backoff_factor=2)
def load_model_with_retry():
    """Load model with retry logic."""
    return TTSModel.load_model()

@retry(max_attempts=3)
def load_voice_with_retry(model, voice):
    """Load voice with retry logic."""
    return model.get_state_for_audio_prompt(voice)

# Usage
try:
    model = load_model_with_retry()
    voice_state = load_voice_with_retry(model, "alba")
    audio = model.generate_audio(voice_state, "Hello!")
    print("Success!")
except Exception as e:
    print(f"Failed after retries: {e}")
```

## Retry Strategies

### Strategy 1: Fixed Delay Retry

```python
import time

def retry_fixed_delay(func, max_attempts=3, delay=2):
    """Retry with fixed delay between attempts."""
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                print(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
                time.sleep(delay)
    raise last_exception

# Usage
def load_voice():
    return model.get_state_for_audio_prompt("remote_voice.wav")

voice_state = retry_fixed_delay(load_voice, max_attempts=3, delay=2)
```

### Strategy 2: Exponential Backoff with Jitter

```python
import time
import random

def retry_exponential_with_jitter(func, max_attempts=5, base_delay=1):
    """Retry with exponential backoff and random jitter."""
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                # Calculate delay with exponential backoff
                delay = base_delay * (2 ** attempt)
                # Add random jitter (±25%)
                jitter = delay * 0.25
                delay = delay + random.uniform(-jitter, jitter)
                print(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s...")
                time.sleep(delay)
    raise last_exception

# Usage
def download_model():
    return TTSModel.load_model()

model = retry_exponential_with_jitter(download_model, max_attempts=5)
```

### Strategy 3: Conditional Retry

```python
def retry_on_specific_errors(func, retryable_errors, max_attempts=3):
    """Retry only on specific errors."""
    last_exception = None
    for attempt in range(max_attempts):
        try:
            return func()
        except tuple(retryable_errors) as e:
            last_exception = e
            if attempt < max_attempts - 1:
                print(f"Retryable error: {e}. Retrying...")
                time.sleep(1)
        except Exception:
            # Don't retry on other errors
            raise
    raise last_exception

# Usage
def load_voice():
    return model.get_state_for_audio_prompt("https://example.com/voice.wav")

# Only retry on network errors
voice_state = retry_on_specific_errors(
    load_voice,
    retryable_errors=[
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    ],
    max_attempts=3
)
```

## Validation Techniques

### Input Validation

```python
import os
from pathlib import Path

def validate_text(text: str, max_length: int = 10000) -> None:
    """Validate input text."""
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    if not text.strip():
        raise ValueError("Text cannot be empty or whitespace")
    if len(text) > max_length:
        raise ValueError(f"Text too long (max {max_length} characters)")

def validate_voice_path(voice_path: str) -> None:
    """Validate voice file path."""
    if not isinstance(voice_path, str):
        raise TypeError("Voice path must be a string")

    # Check if it's a predefined voice name
    predefined_voices = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
    if voice_path in predefined_voices:
        return

    # Check if it's a URL
    if voice_path.startswith(("http://", "https://", "hf://")):
        return

    # Check if it's a local file
    path = Path(voice_path)
    if not path.exists():
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
    if not path.is_file():
        raise ValueError(f"Voice path is not a file: {voice_path}")

    # Check file extension
    valid_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Invalid audio format: {path.suffix}")

def validate_parameters(
    text: str,
    voice: str,
    temperature: float = None,
    decode_steps: int = None
) -> None:
    """Validate all generation parameters."""
    validate_text(text)
    validate_voice_path(voice)

    if temperature is not None:
        if not isinstance(temperature, (int, float)):
            raise TypeError("Temperature must be a number")
        if not 0.0 < temperature <= 2.0:
            raise ValueError("Temperature must be between 0 and 2")

    if decode_steps is not None:
        if not isinstance(decode_steps, int):
            raise TypeError("Decode steps must be an integer")
        if not 1 <= decode_steps <= 32:
            raise ValueError("Decode steps must be between 1 and 32")

# Usage
try:
    validate_parameters(
        text="Hello world!",
        voice="alba",
        temperature=0.7,
        decode_steps=4
    )
    # Proceed with generation
except (TypeError, ValueError, FileNotFoundError) as e:
    print(f"Validation error: {e}")
```

### Resource Validation

```python
import psutil
import torch

def check_system_resources(min_memory_gb=2, min_disk_gb=1):
    """Check if system has sufficient resources."""
    # Check memory
    available_memory = psutil.virtual_memory().available / (1024**3)
    if available_memory < min_memory_gb:
        raise RuntimeError(
            f"Insufficient memory: {available_memory:.1f}GB available, "
            f"{min_memory_gb}GB required"
        )

    # Check disk space (in current directory)
    disk_usage = psutil.disk_usage('.')
    free_disk_gb = disk_usage.free / (1024**3)
    if free_disk_gb < min_disk_gb:
        raise RuntimeError(
            f"Insufficient disk space: {free_disk_gb:.1f}GB available, "
            f"{min_disk_gb}GB required"
        )

    print(f"✅ Resources OK: {available_memory:.1f}GB RAM, {free_disk_gb:.1f}GB disk")

# Usage
try:
    check_system_resources(min_memory_gb=2, min_disk_gb=1)
    model = TTSModel.load_model()
except RuntimeError as e:
    print(f"Resource check failed: {e}")
```

## Graceful Degradation

### Fallback to Default Voice

```python
def generate_with_fallback(text: str, preferred_voice: str = None) -> bytes:
    """Generate speech with voice fallback."""
    model = TTSModel.load_model()

    # Try preferred voice first
    if preferred_voice:
        try:
            voice_state = model.get_state_for_audio_prompt(preferred_voice)
            return model.generate_audio(voice_state, text)
        except Exception as e:
            print(f"Failed to load preferred voice '{preferred_voice}': {e}")
            print("Falling back to default voice...")

    # Fallback to default voice
    try:
        voice_state = model.get_state_for_audio_prompt("alba")
        return model.generate_audio(voice_state, text)
    except Exception as e:
        print(f"Failed to load default voice: {e}")
        raise

# Usage
audio = generate_with_fallback("Hello!", preferred_voice="custom_voice.wav")
```

### Degraded Quality Mode

```python
def generate_with_degraded_mode(text: str, voice: str) -> bytes:
    """Generate speech with degraded quality fallback."""
    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt(voice)

    # Try high quality first
    try:
        audio = model.generate_audio(
            voice_state,
            text,
            # High quality parameters passed via model config
        )
        return audio
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("Out of memory, trying degraded quality mode...")
            # Recreate model with lower quality settings
            model = TTSModel.load_model(
                temp=0.7,
                lsd_decode_steps=1,  # Fewer steps
                noise_clamp=1.0
            )
            voice_state = model.get_state_for_audio_prompt(voice)
            return model.generate_audio(voice_state, text)
        raise

# Usage
audio = generate_with_degraded_mode(long_text, "alba")
```

## Logging and Monitoring

### Structured Logging

```python
import logging
import json
from datetime import datetime

class TTSLogger:
    """Structured logger for TTS operations."""

    def __init__(self, log_file: str = "tts_operations.log"):
        self.logger = logging.getLogger("TTS")
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_operation(
        self,
        operation: str,
        success: bool,
        duration_ms: float = None,
        error: str = None,
        metadata: dict = None
    ):
        """Log TTS operation with structured data."""
        log_data = {
            "operation": operation,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if duration_ms is not None:
            log_data["duration_ms"] = duration_ms
        if error:
            log_data["error"] = error
        if metadata:
            log_data["metadata"] = metadata

        if success:
            self.logger.info(json.dumps(log_data))
        else:
            self.logger.error(json.dumps(log_data))

# Usage
import time

logger = TTSLogger()

def generate_with_logging(text: str, voice: str):
    """Generate speech with comprehensive logging."""
    start_time = time.time()

    try:
        model = TTSModel.load_model()
        voice_state = model.get_state_for_audio_prompt(voice)
        audio = model.generate_audio(voice_state, text)

        duration_ms = (time.time() - start_time) * 1000
        logger.log_operation(
            operation="generate_audio",
            success=True,
            duration_ms=duration_ms,
            metadata={
                "text_length": len(text),
                "voice": voice,
                "audio_samples": audio.shape[0]
            }
        )

        return audio

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.log_operation(
            operation="generate_audio",
            success=False,
            duration_ms=duration_ms,
            error=str(e),
            metadata={"text_length": len(text), "voice": voice}
        )
        raise

# Usage
try:
    audio = generate_with_logging("Hello world!", "alba")
except Exception as e:
    print(f"Generation failed: {e}")
```

### Error Metrics

```python
from collections import defaultdict
from typing import Dict

class TTSErrorTracker:
    """Track and analyze TTS errors."""

    def __init__(self):
        self.errors: Dict[str, int] = defaultdict(int)
        self.total_operations = 0

    def record_error(self, error_type: str):
        """Record an error occurrence."""
        self.errors[error_type] += 1
        self.total_operations += 1

    def record_success(self):
        """Record a successful operation."""
        self.total_operations += 1

    def get_error_rate(self) -> float:
        """Calculate overall error rate."""
        if self.total_operations == 0:
            return 0.0
        error_count = sum(self.errors.values())
        return error_count / self.total_operations

    def get_summary(self) -> dict:
        """Get error summary."""
        return {
            "total_operations": self.total_operations,
            "total_errors": sum(self.errors.values()),
            "error_rate": self.get_error_rate(),
            "errors_by_type": dict(self.errors)
        }

    def print_summary(self):
        """Print error summary."""
        summary = self.get_summary()
        print("\n=== TTS Error Summary ===")
        print(f"Total Operations: {summary['total_operations']}")
        print(f"Total Errors: {summary['total_errors']}")
        print(f"Error Rate: {summary['error_rate']:.2%}")
        print("\nErrors by Type:")
        for error_type, count in summary['errors_by_type'].items():
            print(f"  {error_type}: {count}")

# Usage
tracker = TTSErrorTracker()

def generate_with_tracking(text: str, voice: str):
    """Generate speech with error tracking."""
    try:
        model = TTSModel.load_model()
        voice_state = model.get_state_for_audio_prompt(voice)
        audio = model.generate_audio(voice_state, text)
        tracker.record_success()
        return audio
    except FileNotFoundError as e:
        tracker.record_error("FileNotFoundError")
        raise
    except ValueError as e:
        tracker.record_error("ValueError")
        raise
    except RuntimeError as e:
        tracker.record_error("RuntimeError")
        raise
    except Exception as e:
        tracker.record_error(type(e).__name__)
        raise

# Usage
for i in range(10):
    try:
        generate_with_tracking(f"Test {i}", "alba")
    except Exception:
        pass

tracker.print_summary()
```

## Complete Examples

### Example 1: Robust TTS Service

```python
import logging
from pathlib import Path
from typing import Optional
import torch
import scipy.io.wavfile

class RobustTTSService:
    """Robust TTS service with comprehensive error handling."""

    def __init__(self, compile_model: bool = False):
        self.model = None
        self.voice_cache = {}
        self.compile_model = compile_model
        self._initialize()

    def _initialize(self):
        """Initialize the model with error handling."""
        try:
            self.model = TTSModel.load_model()
            if self.compile_model:
                self.model.compile_for_inference(mode="reduce-overhead")
            logging.info("✅ TTS model initialized successfully")
        except Exception as e:
            logging.error(f"❌ Failed to initialize TTS model: {e}")
            raise RuntimeError(f"TTS initialization failed: {e}")

    def get_voice(self, voice: str) -> dict:
        """Get voice state with caching and error handling."""
        if voice in self.voice_cache:
            return self.voice_cache[voice]

        try:
            voice_state = self.model.get_state_for_audio_prompt(voice)
            self.voice_cache[voice] = voice_state
            return voice_state
        except FileNotFoundError:
            logging.error(f"Voice file not found: {voice}")
            # Try fallback to default
            if voice != "alba":
                logging.info("Falling back to default voice 'alba'")
                return self.get_voice("alba")
            raise
        except Exception as e:
            logging.error(f"Failed to load voice '{voice}': {e}")
            raise

    def generate(
        self,
        text: str,
        voice: str = "alba",
        output_path: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """Generate speech with comprehensive error handling."""
        # Validate inputs
        if not text or not text.strip():
            logging.warning("Empty text provided")
            return None

        try:
            # Get voice state
            voice_state = self.get_voice(voice)

            # Generate audio
            audio = self.model.generate_audio(voice_state, text)

            # Save to file if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                scipy.io.wavfile.write(
                    str(output_path),
                    self.model.sample_rate,
                    audio.numpy()
                )
                logging.info(f"✅ Audio saved to {output_path}")

            return audio

        except Exception as e:
            logging.error(f"❌ Generation failed: {e}")
            return None

    def cleanup(self):
        """Clean up resources."""
        self.voice_cache.clear()
        if self.model:
            del self.model
        import gc
        gc.collect()
        logging.info("✅ Cleanup completed")

# Usage
service = RobustTTSService(compile_model=True)

try:
    audio = service.generate(
        text="Hello, this is a test!",
        voice="alba",
        output_path="output.wav"
    )
    if audio is not None:
        print(f"✅ Generated {audio.shape[0]} samples")
    else:
        print("❌ Generation failed")
finally:
    service.cleanup()
```

### Example 2: Web Service with Error Handling

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
import logging
from pathlib import Path
import uuid

app = FastAPI(title="Robust Pocket TTS Service")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize TTS service
tts_service = RobustTTSService(compile_model=True)

class GenerationRequest(BaseModel):
    text: str
    voice: str = "alba"

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        if len(v) > 10000:
            raise ValueError("Text too long (max 10000 characters)")
        return v

    @validator('voice')
    def validate_voice(cls, v):
        predefined_voices = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
        if v not in predefined_voices and not Path(v).exists():
            raise ValueError(f"Invalid voice: {v}")
        return v

# Store for cleanup
generated_files = {}

@app.post("/generate")
async def generate_audio(request: GenerationRequest):
    """Generate audio with comprehensive error handling."""
    file_id = str(uuid.uuid4())
    output_path = Path(f"/tmp/tts_{file_id}.wav")

    try:
        logger.info(f"Generating audio for text: {request.text[:50]}...")

        # Generate audio
        audio = tts_service.generate(
            text=request.text,
            voice=request.voice,
            output_path=str(output_path)
        )

        if audio is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate audio"
            )

        # Store file for cleanup
        generated_files[file_id] = output_path

        return {
            "status": "success",
            "file_id": file_id,
            "download_url": f"/download/{file_id}",
            "duration_seconds": audio.shape[0] / tts_service.model.sample_rate
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{file_id}")
async def download_audio(file_id: str):
    """Download generated audio file."""
    if file_id not in generated_files:
        raise HTTPException(status_code=404, detail="File not found")

    output_path = generated_files[file_id]
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename=f"tts_{file_id}.wav"
    )

def cleanup_file(file_id: str):
    """Clean up generated file."""
    if file_id in generated_files:
        file_path = generated_files[file_id]
        try:
            file_path.unlink()
            del generated_files[file_id]
            logger.info(f"Cleaned up file: {file_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup file {file_id}: {e}")

@app.post("/cleanup/{file_id}")
async def cleanup_endpoint(file_id: str, background_tasks: BackgroundTasks):
    """Clean up a generated file."""
    background_tasks.add_task(cleanup_file, file_id)
    return {"status": "cleanup_scheduled"}

# Run with: uvicorn tts_service:app --host 0.0.0.0 --port 8000
```

For more information, see:
- [Python API Documentation](python-api.md)
- [Configuration Guide](configuration-guide.md)
- [Troubleshooting Guide](troubleshooting.md)
