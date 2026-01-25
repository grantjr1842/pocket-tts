# Migration Guide

This guide helps users migrate between different versions of Pocket TTS and adapt their code to new APIs and features.

## Version History and Changes

### Version 1.0 → 1.1

#### Breaking Changes
- **TTSModel.load_model()** signature changed:
  ```python
  # Old (v1.0)
  model = TTSModel.load_model("model_path")
  
  # New (v1.1+)
  model = TTSModel.load_model(variant="b6369a24")
  ```

- **Voice loading** method renamed:
  ```python
  # Old (v1.0)
  voice_embeddings = model.load_voice("voice.wav")
  
  # New (v1.1+)
  voice_state = model.get_state_for_audio_prompt("voice.wav")
  ```

#### Migration Steps
```python
# Step 1: Update model loading
try:
    # Try new API first
    model = TTSModel.load_model(variant="b6369a24")
except TypeError:
    # Fallback to old API for compatibility
    model = TTSModel.load_model("b6369a24")

# Step 2: Update voice loading
try:
    # New API
    voice_state = model.get_state_for_audio_prompt("alba")
except AttributeError:
    # Fallback to old API
    voice_embeddings = model.load_voice("alba")
    voice_state = voice_embeddings  # Adapt to new variable name
```

### Version 1.1 → 1.2

#### New Features
- **Streaming generation**: `generate_audio_stream()` method added
- **Compilation support**: `compile_for_inference()` method added
- **Rust acceleration**: Optional Rust extensions for audio processing

#### Migration Steps
```python
# Step 1: Update imports if using Rust extensions
try:
    from pocket_tts import (
        normalize_audio, 
        apply_gain, 
        resample_audio,
        _RUST_NUMPY_AVAILABLE
    )
    rust_available = _RUST_NUMPY_AVAILABLE
except ImportError:
    from pocket_tts import _RUST_NUMPY_AVAILABLE
    rust_available = False

# Step 2: Enable compilation for better performance
model = TTSModel.load_model()
try:
    model.compile_for_inference()
    print("Compilation enabled successfully")
except AttributeError:
    print("Compilation not available in this version")

# Step 3: Use streaming for long texts
def generate_long_text(model, voice_state, text):
    """Generate long text with streaming if available."""
    try:
        # Try new streaming API
        audio_chunks = []
        for chunk in model.generate_audio_stream(voice_state, text):
            audio_chunks.append(chunk)
        return torch.cat(audio_chunks, dim=0)
    except AttributeError:
        # Fallback to old batch generation
        return model.generate_audio(voice_state, text)
```

### Version 1.2 → 1.3

#### Breaking Changes
- **Error handling**: More specific exceptions introduced
- **Audio format**: Standardized to 24kHz, 16-bit PCM
- **Parameter validation**: stricter input validation

#### Migration Steps
```python
# Step 1: Update error handling
try:
    model = TTSModel.load_model()
except ValueError as e:
    # Handle specific parameter errors
    if "temperature" in str(e):
        print("Temperature parameter out of range (0.1-2.0)")
    elif "variant" in str(e):
        print("Invalid model variant specified")
    else:
        print(f"Parameter error: {e}")

# Step 2: Update audio processing
from pocket_tts import save_audio

# Old code (may need updating)
# scipy.io.wavfile.write("output.wav", sample_rate, audio.numpy())

# New recommended code
save_audio("output.wav", audio, sample_rate)  # Handles format automatically
```

## API Migration Patterns

### General Migration Strategy

```python
class VersionAdapter:
    """Adapter pattern for handling multiple API versions."""
    
    def __init__(self, target_version="1.3"):
        self.target_version = target_version
        self.model = self._load_model()
        self.api_version = self._detect_api_version()
    
    def _load_model(self):
        """Load model using appropriate API for detected version."""
        try:
            # Try newest API
            if self.target_version >= "1.3":
                return TTSModel.load_model(variant="b6369a24")
            elif self.target_version >= "1.2":
                return TTSModel.load_model("b6369a24")
            else:
                return TTSModel.load_model()
                
        except Exception as e:
            print(f"Failed to load with version {self.target_version} API: {e}")
            # Try with oldest compatible API
            return TTSModel.load_model()
    
    def _detect_api_version(self):
        """Detect which API version is available."""
        model = TTSModel.load_model()
        
        # Check for newer methods
        if hasattr(model, 'compile_for_inference'):
            return "1.3"
        elif hasattr(model, 'generate_audio_stream'):
            return "1.2"
        else:
            return "1.1"
    
    def get_voice_state(self, voice_input):
        """Get voice state using available API."""
        if self.api_version >= "1.1":
            return self.model.get_state_for_audio_prompt(voice_input)
        else:
            # Fallback to older method if it exists
            if hasattr(self.model, 'load_voice'):
                return self.model.load_voice(voice_input)
            else:
                # Default implementation for very old versions
                raise NotImplementedError("Voice loading not supported in this version")
    
    def generate_audio(self, voice, text, use_streaming=False):
        """Generate audio using best available method."""
        voice_state = self.get_voice_state(voice)
        
        if use_streaming and self.api_version >= "1.2":
            # Use streaming for better memory usage
            chunks = []
            for chunk in self.model.generate_audio_stream(voice_state, text):
                chunks.append(chunk)
            return torch.cat(chunks, dim=0)
        else:
            # Use batch generation
            return self.model.generate_audio(voice_state, text)

# Usage
adapter = VersionAdapter(target_version="1.3")
audio = adapter.generate_audio("alba", "Hello world", use_streaming=True)
```

### Configuration Migration

```python
def migrate_configuration(old_config_path, new_config_path):
    """Migrate configuration from old to new format."""
    
    # Load old configuration
    old_config = {}
    if os.path.exists(old_config_path):
        with open(old_config_path) as f:
            old_config = json.load(f)
    
    # Map old parameters to new ones
    new_config = {
        # Direct mappings
        "model_variant": old_config.get("model_name", "b6369a24"),
        "generation_temperature": old_config.get("temp", 0.7),
        "decode_steps": old_config.get("steps", 1),
        "compilation_enabled": old_config.get("compile", False),
        
        # New defaults
        "noise_clamp": old_config.get("noise_limit", None),
        "eos_threshold": old_config.get("stop_threshold", -4.0),
        "frames_after_eos": old_config.get("tail_frames", None),
    }
    
    # Save new configuration
    with open(new_config_path, 'w') as f:
        json.dump(new_config, f, indent=2)
    
    print(f"Configuration migrated from {old_config_path} to {new_config_path}")
    return new_config

# Usage
old_config = "./old_config.json"
new_config = "./config.json"

if os.path.exists(old_config):
    migrate_configuration(old_config, new_config)
```

## Code Migration Examples

### Migrating Simple CLI Applications

```python
# Old code (v1.0)
import sys
from pocket_tts import TTSModel

def old_cli_app(text, voice_file):
    model = TTSModel.load_model()
    voice_embeddings = model.load_voice(voice_file)
    audio = model.generate_audio(voice_embeddings, text)
    scipy.io.wavfile.write("output.wav", 24000, audio.numpy())

# Migrated code (v1.3+)
import sys
from pocket_tts import TTSModel, save_audio

def new_cli_app(text, voice="alba"):
    # Load with new API
    model = TTSModel.load_model(variant="b6369a24")
    model.compile_for_inference()  # Enable performance optimizations
    
    # Get voice state with new API
    voice_state = model.get_state_for_audio_prompt(voice)
    
    # Generate audio
    audio = model.generate_audio(voice_state, text)
    
    # Save with new API
    save_audio("output.wav", audio, model.sample_rate)

# Compatibility wrapper that works with both versions
def compatible_cli_app(text, voice_or_file):
    try:
        # Try new API first
        new_cli_app(text, voice_or_file)
    except (AttributeError, TypeError):
        # Fallback to old API
        old_cli_app(text, voice_or_file)
```

### Migrating Web Applications

```python
# Old Flask application (v1.0)
from flask import Flask, request, jsonify
from pocket_tts import TTSModel

app = Flask(__name__)
model = TTSModel.load_model()

@app.route('/generate', methods=['POST'])
def old_generate():
    data = request.json
    voice_file = data.get('voice_file')
    text = data.get('text')
    
    voice_embeddings = model.load_voice(voice_file)
    audio = model.generate_audio(voice_embeddings, text)
    
    return jsonify({
        'audio': audio.tolist(),
        'sample_rate': 24000
    })

# Migrated FastAPI application (v1.3+)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pocket_tts import TTSModel, save_audio

class GenerateRequest(BaseModel):
    text: str
    voice: str = "alba"
    temperature: float = 0.7

app = FastAPI()

# Initialize with new API
model = TTSModel.load_model(variant="b6369a24")
model.compile_for_inference()

@app.post('/generate')
async def new_generate(request: GenerateRequest):
    try:
        voice_state = model.get_state_for_audio_prompt(request.voice)
        audio = model.generate_audio(
            voice_state, 
            request.text,
            temperature=request.temperature
        )
        
        return {
            'audio': audio.tolist(),
            'sample_rate': model.sample_rate,
            'format': 'wav'
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/generate-stream')
async def stream_generate(request: GenerateRequest):
    """New streaming endpoint for better performance."""
    try:
        voice_state = model.get_state_for_audio_prompt(request.voice)
        
        async def audio_stream():
            for chunk in model.generate_audio_stream(voice_state, request.text):
                yield chunk.numpy().tobytes()
        
        return StreamingResponse(
            audio_stream(),
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Dependency Migration

### Updating Requirements

```txt
# Old requirements.txt (v1.0)
torch>=1.9.0
torchaudio
numpy
scipy

# New requirements.txt (v1.3+)
torch>=2.5.0
torchaudio>=2.5.0
numpy>=1.21.0
scipy>=1.7.0
safetensors>=0.3.0
beartype>=0.16.0

# Optional Rust extensions
# Add these for better performance:
# pip install "pocket-tts[rust]"  # If available
# Or build manually:
# cd training/rust_exts/audio_ds && cargo build --release
```

### Docker Migration

```dockerfile
# Old Dockerfile (v1.0)
FROM python:3.9-slim

RUN pip install torch torchvision torchaudio
COPY . .
RUN pip install .

# New Dockerfile (v1.3+)
FROM python:3.11-slim

# Install system dependencies for Rust extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    rustc \
    cargo

# Install updated dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application
COPY . .

# Health check for new API
RUN python -c "from pocket_tts import TTSModel; TTSModel.load_model()" || exit 1

EXPOSE 8000
CMD ["python", "main.py", "serve"]
```

## Testing Migration

### Version Compatibility Tests

```python
import pytest
import tempfile
from pocket_tts import TTSModel

class TestMigration:
    def __init__(self):
        self.model = None
    
    def test_api_compatibility(self):
        """Test if current API works with expected methods."""
        try:
            self.model = TTSModel.load_model()
            
            # Test core methods exist
            assert hasattr(self.model, 'generate_audio'), "generate_audio method missing"
            assert hasattr(self.model, 'sample_rate'), "sample_rate property missing"
            
            # Test newer methods if available
            if hasattr(self.model, 'generate_audio_stream'):
                print("✅ Streaming API available")
            if hasattr(self.model, 'compile_for_inference'):
                print("✅ Compilation API available")
            
            return True
            
        except Exception as e:
            print(f"❌ API compatibility test failed: {e}")
            return False
    
    def test_voice_loading(self):
        """Test voice loading with current API."""
        try:
            # Test with preset voice
            voice_state = self.model.get_state_for_audio_prompt("alba")
            assert voice_state is not None, "Failed to load preset voice"
            
            # Test with file path
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                # Create a dummy audio file
                import torch
                dummy_audio = torch.randn(24000)  # 1 second
                from pocket_tts import save_audio
                save_audio(tmp.name, dummy_audio, 24000)
                
                voice_state = self.model.get_state_for_audio_prompt(tmp.name)
                assert voice_state is not None, "Failed to load file voice"
            
            print("✅ Voice loading compatibility test passed")
            return True
            
        except Exception as e:
            print(f"❌ Voice loading test failed: {e}")
            return False
    
    def test_generation(self):
        """Test audio generation."""
        try:
            voice_state = self.model.get_state_for_audio_prompt("alba")
            audio = self.model.generate_audio(voice_state, "Test migration compatibility")
            
            assert audio.shape[0] > 0, "Generated audio is empty"
            assert audio.dtype == torch.float32, "Audio has wrong data type"
            
            print("✅ Generation compatibility test passed")
            return True
            
        except Exception as e:
            print(f"❌ Generation test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all compatibility tests."""
        tests = [
            ("API Compatibility", self.test_api_compatibility),
            ("Voice Loading", self.test_voice_loading),
            ("Audio Generation", self.test_generation),
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\nRunning {test_name} test...")
            results[test_name] = test_func()
        
        print(f"\nMigration Test Results:")
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        all_passed = all(results.values())
        if all_passed:
            print("\n🎉 All migration tests passed!")
        else:
            print("\n⚠️  Some tests failed. Check API compatibility.")
        
        return all_passed

# Usage
if __name__ == "__main__":
    migration_tests = TestMigration()
    migration_tests.run_all_tests()
```

## Rollback Procedures

### Quick Rollback Script

```python
#!/usr/bin/env python3
"""
Pocket TTS Rollback Script
Revert to previous version if migration causes issues.
"""

import subprocess
import sys
import os

def rollback_to_version(version):
    """Rollback to specific version."""
    print(f"Rolling back to Pocket TTS version {version}...")
    
    # Uninstall current version
    print("Uninstalling current version...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "pocket-tts", "-y"], 
                  check=True)
    
    # Install specific version
    print(f"Installing Pocket TTS {version}...")
    subprocess.run([sys.executable, "-m", "pip", "install", 
                  f"pocket-tts=={version}"], check=True)
    
    # Verify installation
    try:
        from pocket_tts import TTSModel
        model = TTSModel.load_model()
        print(f"✅ Successfully rolled back to version {version}")
        return True
    except Exception as e:
        print(f"❌ Rollback failed: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python rollback.py <version>")
        print("Example: python rollback.py 1.2.0")
        sys.exit(1)
    
    version = sys.argv[1]
    rollback_to_version(version)

if __name__ == "__main__":
    main()
```

### Configuration Backup and Restore

```python
import json
import shutil
from datetime import datetime

def backup_configuration(config_path="./config.json"):
    """Backup current configuration."""
    if not os.path.exists(config_path):
        print("No configuration file to backup")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{config_path}.backup_{timestamp}"
    
    shutil.copy2(config_path, backup_path)
    print(f"Configuration backed up to: {backup_path}")
    return backup_path

def restore_configuration(backup_path):
    """Restore configuration from backup."""
    if not os.path.exists(backup_path):
        print(f"Backup file not found: {backup_path}")
        return
    
    config_path = "./config.json"
    shutil.copy2(backup_path, config_path)
    print(f"Configuration restored from: {backup_path}")
    return config_path

# Usage
backup_path = backup_configuration()
# If migration fails:
restore_configuration(backup_path)
```

This migration guide should help you smoothly transition between different versions of Pocket TTS while maintaining functionality and taking advantage of new features.