# Error Handling Guide

This guide covers proper error handling patterns for Pocket TTS, including common exceptions, troubleshooting, and best practices.

## Exception Types

### Core Exceptions

#### `ValueError`
Raised when input parameters are invalid or out of expected ranges.

```python
from pocket_tts import TTSModel

try:
    model = TTSModel.load_model(temperature=5.0)  # Invalid: > 2.0
except ValueError as e:
    print(f"Parameter error: {e}")
    # Handle: Use valid temperature range (0.1-2.0)
    model = TTSModel.load_model(temperature=0.7)
```

#### `FileNotFoundError`
Raised when audio files or model weights cannot be found.

```python
from pocket_tts import TTSModel, load_wav

try:
    # Missing voice file
    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt("./missing_voice.wav")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    # Handle: Use default voice or check file path
    voice_state = model.get_state_for_audio_prompt("alba")
```

#### `OSError`
Raised when system operations fail (disk space, permissions, etc.).

```python
from pocket_tts import save_audio
import torch

audio = torch.randn(24000)  # 1 second of audio

try:
    save_audio("/protected/path/output.wav", audio, 24000)
except OSError as e:
    print(f"System error: {e}")
    # Handle: Check permissions, disk space, or use different path
    save_audio("./output.wav", audio, 24000)
```

#### `ImportError`
Raised when optional dependencies are missing.

```python
try:
    from pocket_tts import _RUST_NUMPY_AVAILABLE
    if _RUST_NUMPY_AVAILABLE:
        print("Rust acceleration enabled")
    else:
        print("Using Python fallback")
except ImportError as e:
    print(f"Import error: {e}")
    # Handle: Install missing dependencies
    # pip install pocket-tts[rust]  # if available
```

#### `TypeError`
Raised when data types are incorrect or incompatible.

```python
from pocket_tts import save_audio
import numpy as np

try:
    # Wrong data type
    save_audio("output.wav", "not_audio_data", 24000)
except TypeError as e:
    print(f"Type error: {e}")
    # Handle: Ensure audio is tensor or array
    audio = np.random.randn(24000).astype(np.float32)
    save_audio("output.wav", audio, 24000)
```

## Common Error Scenarios

### 1. Model Loading Errors

#### Out of Memory
```python
import torch
from pocket_tts import TTSModel

try:
    model = TTSModel.load_model()
except torch.cuda.OutOfMemoryError:
    print("GPU out of memory, falling back to CPU")
    model = TTSModel.load_model(device="cpu")
except Exception as e:
    print(f"Model loading failed: {e}")
    # Try clearing cache
    import torch
    torch.cuda.empty_cache()
    model = TTSModel.load_model()
```

#### Network Issues
```python
import requests
from pocket_tts import TTSModel

def load_model_with_retry(max_retries=3):
    for attempt in range(max_retries):
        try:
            model = TTSModel.load_model()
            return model
        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"Network error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
    
    raise Exception("Failed to load model after retries")

model = load_model_with_retry()
```

### 2. Audio Processing Errors

#### Invalid Audio Format
```python
from pocket_tts import load_wav, get_state_for_audio_prompt

def safe_load_audio(audio_path):
    try:
        audio = load_wav(audio_path)
        
        # Validate audio properties
        if len(audio.shape) != 1:
            raise ValueError(f"Expected mono audio, got {len(audio.shape)} channels")
        
        if audio.max() > 1.0 or audio.min() < -1.0:
            print("Warning: Audio values outside [-1, 1] range, normalizing...")
            audio = torch.clamp(audio, -1.0, 1.0)
        
        return audio
        
    except Exception as e:
        print(f"Audio loading error: {e}")
        # Return default audio or raise
        raise ValueError(f"Could not load audio from {audio_path}: {e}")

# Usage
try:
    audio = safe_load_audio("./voice.wav")
    voice_state = model.get_state_for_audio_prompt(audio)
except ValueError as e:
    print(f"Using default voice due to audio error: {e}")
    voice_state = model.get_state_for_audio_prompt("alba")
```

#### Empty or Corrupted Audio
```python
def validate_audio(audio, sample_rate):
    """Validate audio data and return cleaned version or raise error."""
    
    if audio is None or len(audio) == 0:
        raise ValueError("Audio data is empty")
    
    if not isinstance(audio, (torch.Tensor, np.ndarray)):
        raise TypeError(f"Audio must be tensor or array, got {type(audio)}")
    
    # Convert to tensor if needed
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    
    # Ensure single channel
    if len(audio.shape) > 1:
        audio = audio.mean(dim=0)
    
    # Check for NaN or infinite values
    if torch.isnan(audio).any() or torch.isinf(audio).any():
        print("Warning: Audio contains NaN or infinite values, replacing with zeros")
        audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return audio

# Usage
try:
    audio = load_wav("./voice.wav")
    audio = validate_audio(audio, 24000)
    voice_state = model.get_state_for_audio_prompt(audio)
except (ValueError, TypeError) as e:
    print(f"Audio validation failed: {e}")
    voice_state = model.get_state_for_audio_prompt("alba")
```

### 3. Generation Errors

#### Text Processing Issues
```python
def safe_generate_audio(model, voice_state, text):
    """Generate audio with proper error handling."""
    
    # Validate text
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    if len(text) > 10000:  # Reasonable limit
        print(f"Warning: Long text ({len(text)} chars) may cause memory issues")
    
    try:
        return model.generate_audio(voice_state, text)
    except torch.cuda.OutOfMemoryError:
        print("GPU memory error, trying CPU...")
        cpu_model = TTSModel.load_model(device="cpu")
        cpu_voice_state = cpu_model.get_state_for_audio_prompt("alba")
        return cpu_model.generate_audio(cpu_voice_state, text)
    except Exception as e:
        print(f"Generation failed: {e}")
        # Try with conservative settings
        try:
            conservative_model = TTSModel.load_model(
                temperature=0.5,
                lsd_decode_steps=1
            )
            return conservative_model.generate_audio(voice_state, text)
        except Exception as e2:
            raise RuntimeError(f"Failed to generate audio even with fallback: {e2}")

# Usage
try:
    audio = safe_generate_audio(model, voice_state, "Hello world")
    save_audio("output.wav", audio, model.sample_rate)
except Exception as e:
    print(f"Audio generation completely failed: {e}")
```

## Server Error Handling

### WebSocket Connection Issues
```python
from fastapi import WebSocket, WebSocketDisconnect
import json

async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        
        while True:
            # Receive message with timeout
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(), 
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue
            
            try:
                data = json.loads(message)
                text = data.get("text", "")
                voice = data.get("voice", "alba")
                
                # Generate audio
                voice_state = model.get_state_for_audio_prompt(voice)
                audio = model.generate_audio(voice_state, text)
                
                # Send back
                await websocket.send_json({
                    "type": "audio",
                    "data": audio.tolist(),
                    "sample_rate": model.sample_rate
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()
```

### HTTP API Error Handling
```python
from fastapi import HTTPException
import traceback

@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    try:
        # Validate request
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long (max 5000 chars)")
        
        # Generate audio
        voice_state = model.get_state_for_audio_prompt(request.voice or "alba")
        audio = model.generate_audio(voice_state, request.text)
        
        return {"audio": audio.tolist(), "sample_rate": model.sample_rate}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Unexpected error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

## Best Practices

### 1. Graceful Degradation
```python
def robust_tts_generation(text, voice="alba"):
    """TTS generation with multiple fallback levels."""
    
    # Level 1: Try with requested voice
    try:
        model = TTSModel.load_model()
        voice_state = model.get_state_for_audio_prompt(voice)
        return model.generate_audio(voice_state, text)
    except Exception as e:
        print(f"Level 1 failed: {e}")
    
    # Level 2: Try with default voice
    try:
        voice_state = model.get_state_for_audio_prompt("alba")
        return model.generate_audio(voice_state, text)
    except Exception as e:
        print(f"Level 2 failed: {e}")
    
    # Level 3: Try with conservative settings
    try:
        model = TTSModel.load_model(temperature=0.5, lsd_decode_steps=1)
        voice_state = model.get_state_for_audio_prompt("alba")
        return model.generate_audio(voice_state, text)
    except Exception as e:
        print(f"Level 3 failed: {e}")
    
    # All failed
    raise RuntimeError("Unable to generate audio with any fallback")
```

### 2. Resource Management
```python
import contextlib
import tempfile
import os

@contextlib.contextmanager
def temporary_model():
    """Context manager for temporary model instances."""
    model = None
    try:
        model = TTSModel.load_model()
        yield model
    finally:
        # Cleanup if needed
        if model is not None:
            del model
            import torch
            torch.cuda.empty_cache()

# Usage
with temporary_model() as model:
    voice_state = model.get_state_for_audio_prompt("alba")
    audio = model.generate_audio(voice_state, "Hello world")
    # Model automatically cleaned up
```

### 3. Logging and Monitoring
```python
import logging
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pocket_tts")

def monitor_performance(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
    return wrapper

# Usage
@monitor_performance
def generate_with_monitoring(model, voice_state, text):
    return model.generate_audio(voice_state, text)
```

## Debug Mode

### Verbose Error Information
```python
import sys
import traceback

def enable_debug_mode():
    """Enable detailed error reporting."""
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Global exception handler
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))
        traceback.print_exception(exc_type, exc_value, exc_traceback)
    
    sys.excepthook = handle_exception

# Enable in development
if __name__ == "__main__":
    enable_debug_mode()
    # Your application code here
```

This comprehensive error handling guide ensures your Pocket TTS applications are robust, user-friendly, and provide meaningful feedback when issues occur.