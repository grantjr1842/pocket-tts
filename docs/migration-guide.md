# Pocket TTS Migration Guide

This guide helps you migrate between different versions of Pocket TTS, understand breaking changes, and implement rollback procedures.

## Table of Contents

- [Version Compatibility](#version-compatibility)
- [Breaking Changes](#breaking-changes)
- [Migration Procedures](#migration-procedures)
- [Rollback Procedures](#rollback-procedures)
- [Deprecation Notices](#deprecation-notices)
- [Testing Migrations](#testing-migrations)

## Version Compatibility

### Python Version Compatibility

| Pocket TTS Version | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 | Python 3.14 |
|-------------------|-------------|-------------|-------------|-------------|-------------|
| 0.1.x             | ✅          | ✅          | ✅          | ✅          | ❌          |
| 0.2.x             | ✅          | ✅          | ✅          | ✅          | ✅          |
| 1.0.x             | ✅          | ✅          | ✅          | ✅          | ✅          |

### PyTorch Version Requirements

| Pocket TTS Version | Minimum PyTorch | Recommended PyTorch |
|-------------------|-----------------|---------------------|
| 0.1.x             | 2.3.0           | 2.3.1               |
| 0.2.x             | 2.4.0           | 2.4.1               |
| 1.0.x             | 2.5.0           | 2.5.1               |

### Platform Compatibility

All versions support:
- **Linux**: x86_64, ARM64 (aarch64)
- **macOS**: x86_64, Apple Silicon (M1/M2/M3/M4)
- **Windows**: x86_64

## Breaking Changes

### Version 1.0.0 → Current

#### API Changes

**Before (0.2.x):**
```python
from pocket_tts import TTSModel

model = TTSModel(
    variant="b6369a24",
    temp=0.7,
    steps=4
)
```

**After (1.0.x):**
```python
from pocket_tts import TTSModel

model = TTSModel.load_model(
    variant="b6369a24",
    temp=0.7,
    lsd_decode_steps=4  # Renamed parameter
)
```

**Migration:** Replace `TTSModel()` constructor with `TTSModel.load_model()` class method, and rename `steps` to `lsd_decode_steps`.

#### Parameter Name Changes

| Old Parameter (0.2.x) | New Parameter (1.0.x) |
|-----------------------|------------------------|
| `steps`               | `lsd_decode_steps`     |
| `eos`                 | `eos_threshold`        |

**Migration Script:**
```python
import re

def migrate_code_v02_to_v10(code_file):
    """Migrate code from 0.2.x to 1.0.x."""
    with open(code_file, 'r') as f:
        code = f.read()

    # Replace constructor
    code = re.sub(
        r'TTSModel\(',
        'TTSModel.load_model(',
        code
    )

    # Rename parameter
    code = re.sub(
        r'\bsteps\s*=',
        'lsd_decode_steps=',
        code
    )

    # Rename eos parameter
    code = re.sub(
        r'\beos\s*=',
        'eos_threshold=',
        code
    )

    with open(code_file, 'w') as f:
        f.write(code)

    print(f"Migrated {code_file}")

# Usage
migrate_code_v02_to_v10("my_app.py")
```

### Version 0.2.x → 1.0.0

#### Voice Loading Changes

**Before (0.2.x):**
```python
voice_state = model.load_voice("alba")
audio = model.generate(voice_state, "Hello world")
```

**After (1.0.x):**
```python
voice_state = model.get_state_for_audio_prompt("alba")
audio = model.generate_audio(voice_state, "Hello world")
```

#### Method Renames

| Old Method (0.2.x) | New Method (1.0.x) |
|--------------------|--------------------|
| `load_voice()`     | `get_state_for_audio_prompt()` |
| `generate()`       | `generate_audio()`  |
| `stream()`         | `generate_audio_stream()` |

**Migration Script:**
```python
import re

def migrate_method_names(code_file):
    """Migrate method names from 0.2.x to 1.0.x."""
    with open(code_file, 'r') as f:
        code = f.read()

    replacements = [
        (r'\.load_voice\(', '.get_state_for_audio_prompt('),
        (r'\.generate\(', '.generate_audio('),
        (r'\.stream\(', '.generate_audio_stream('),
    ]

    for old, new in replacements:
        code = re.sub(old, new, code)

    with open(code_file, 'w') as f:
        f.write(code)

    print(f"Migrated methods in {code_file}")
```

## Migration Procedures

### Pre-Migration Checklist

Before migrating to a new version:

1. **Backup your code**: Create a git commit or branch
2. **Check compatibility**: Verify Python and PyTorch versions
3. **Review breaking changes**: Read the release notes carefully
4. **Test in isolation**: Use a virtual environment for testing
5. **Plan rollback**: Know how to revert if needed

### Step-by-Step Migration

#### Step 1: Create a Migration Branch

```bash
# Create branch for migration
git checkout -b migrate-to-v1.0

# Or for GitHub Ralph Loop
git worktree add -b migrate-to-v1.0 ../worktree-migrate
cd ../worktree-migrate
```

#### Step 2: Update Dependencies

```bash
# Create virtual environment for testing
python -m venv venv-test
source venv-test/bin/activate  # On Windows: venv-test\Scripts\activate

# Update Pocket TTS
pip install --upgrade 'pocket-tts>=1.0.0'

# Verify installation
pip show pocket-tts
python -c "from pocket_tts import TTSModel; print(TTSModel.__version__)"
```

#### Step 3: Update Code

**Automatic migration:**
```python
import ast
import re

class Migrator(ast.NodeTransformer):
    """AST-based migrator for Pocket TTS code."""

    def visit_Call(self, node):
        # Migrate TTSModel() -> TTSModel.load_model()
        if isinstance(node.func, ast.Name) and node.func.id == 'TTSModel':
            node.func = ast.Attribute(
                value=ast.Name(id='TTSModel', ctx=ast.Load()),
                attr='load_model',
                ctx=ast.Load()
            )

        # Migrate method names
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'load_voice':
                node.func.attr = 'get_state_for_audio_prompt'
            elif node.func.attr == 'generate':
                node.func.attr = 'generate_audio'
            elif node.func.attr == 'stream':
                node.func.attr = 'generate_audio_stream'

        return self.generic_visit(node)

def migrate_file(input_file, output_file=None):
    """Migrate a Python file to new API."""
    with open(input_file, 'r') as f:
        code = f.read()

    try:
        tree = ast.parse(code)
        migrator = Migrator()
        new_tree = migrator.visit(tree)
        new_code = ast.unparse(new_tree)

        output_file = output_file or input_file
        with open(output_file, 'w') as f:
            f.write(new_code)

        print(f"Migrated {input_file}")
        return True

    except Exception as e:
        print(f"Error migrating {input_file}: {e}")
        return False

# Migrate all Python files in directory
import os

for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            migrate_file(filepath)
```

**Manual migration checklist:**
- [ ] Replace `TTSModel()` with `TTSModel.load_model()`
- [ ] Rename `steps` parameter to `lsd_decode_steps`
- [ ] Rename `eos` parameter to `eos_threshold`
- [ ] Replace `load_voice()` with `get_state_for_audio_prompt()`
- [ ] Replace `generate()` with `generate_audio()`
- [ ] Replace `stream()` with `generate_audio_stream()`

#### Step 4: Test Your Changes

```python
# test_migration.py
import sys

def test_basic_generation():
    """Test basic TTS generation."""
    from pocket_tts import TTSModel

    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt("alba")
    audio = model.generate_audio(voice_state, "Test")

    assert audio is not None
    assert audio.shape[0] > 0
    print("✅ Basic generation test passed")

def test_streaming():
    """Test audio streaming."""
    from pocket_tts import TTSModel

    model = TTSModel.load_model()
    voice_state = model.get_state_for_audio_prompt("alba")

    chunks = list(model.generate_audio_stream(voice_state, "Test streaming"))
    assert len(chunks) > 0
    print("✅ Streaming test passed")

def test_parameters():
    """Test new parameter names."""
    from pocket_tts import TTSModel

    model = TTSModel.load_model(
        temp=0.7,
        lsd_decode_steps=4,
        eos_threshold=-4.0
    )

    voice_state = model.get_state_for_audio_prompt("alba")
    audio = model.generate_audio(voice_state, "Test parameters")

    assert audio is not None
    print("✅ Parameter test passed")

if __name__ == "__main__":
    try:
        test_basic_generation()
        test_streaming()
        test_parameters()
        print("\n✅ All migration tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Migration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

Run tests:
```bash
python test_migration.py
```

#### Step 5: Update Deployment

**Docker:**
```dockerfile
# Update Pocket TTS version
FROM python:3.11-slim

RUN pip install --no-cache-dir 'pocket-tts>=1.0.0'

# Rest of your Dockerfile...
```

**requirements.txt:**
```txt
# Update version constraint
pocket-tts>=1.0.0,<2.0.0
```

**GitHub Actions:**
```yaml
# .github/workflows/test.yml
- name: Install dependencies
  run: |
    pip install 'pocket-tts>=1.0.0'
```

## Rollback Procedures

### Immediate Rollback

#### Revert Code Changes

```bash
# If using git
git checkout main
git branch -D migrate-to-v1.0

# Or revert commits
git revert <commit-hash>
git push
```

#### Downgrade Package

```bash
# Uninstall new version
pip uninstall pocket-tts

# Install previous version
pip install 'pocket-tts==0.2.0'

# Or using version constraints
pip install 'pocket-tts<1.0.0'
```

### Rollback Strategy

#### Option 1: Virtual Environment Swap

```bash
# Keep two environments
python -m venv venv-old  # Previous version
python -m venv venv-new  # New version

# If new version fails, switch to old
deactivate  # Exit new environment
source venv-old/bin/activate  # Activate old environment
```

#### Option 2: Docker Image Rollback

```bash
# Build with version tag
docker build -t pocket-tts:1.0 .
docker build -t pocket-tts:0.2 .

# Run specific version
docker run pocket-tts:0.2

# If 1.0 fails, quickly switch
docker stop pocket-tts-1.0
docker run -d --name pocket-tts-rollback pocket-tts:0.2
```

#### Option 3: Feature Flags

```python
import os

USE_NEW_VERSION = os.environ.get('USE_TTS_V1', 'false').lower() == 'true'

if USE_NEW_VERSION:
    from pocket_tts_v1 import TTSModel
else:
    from pocket_tts_v0 import TTSModel
```

## Deprecation Notices

### Deprecated in Version 1.0

The following features are deprecated and will be removed in version 2.0:

#### Direct Model Construction

**Deprecated:**
```python
model = TTSModel.load_model()  # Current method
```

**Will be removed in 2.0:** None (this is the new pattern)

Old pattern was already removed in 1.0:
```python
model = TTSModel()  # ❌ Removed in 1.0
```

#### Old Parameter Names

The following parameter names are deprecated:

- `steps` → Use `lsd_decode_steps` instead
- `eos` → Use `eos_threshold` instead

**Migration timeline:**
- **Version 1.0**: Old names still work with deprecation warning
- **Version 1.5**: Old names emit warnings
- **Version 2.0**: Old names removed

### Upcoming Changes

#### Version 2.0 Planning

Planned changes for version 2.0:

1. **Async API**: Introduction of async methods for better concurrency
2. **Streaming improvements**: Enhanced streaming API with callbacks
3. **Multiple model support**: Support for loading multiple model variants

**Prepare for migration:**
```python
# Current (1.x)
model = TTSModel.load_model()
audio = model.generate_audio(voice_state, text)

# Future (2.x) - async
model = await TTSModel.load_model_async()
audio = await model.generate_audio_async(voice_state, text)
```

## Testing Migrations

### Unit Tests

```python
# test_migration.py
import unittest
from pocket_tts import TTSModel

class TestMigration(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.model = TTSModel.load_model()
        self.voice_state = self.model.get_state_for_audio_prompt("alba")

    def test_api_changes(self):
        """Test new API methods exist."""
        # Check new methods exist
        self.assertTrue(hasattr(self.model, 'generate_audio'))
        self.assertTrue(hasattr(self.model, 'generate_audio_stream'))
        self.assertTrue(hasattr(self.model, 'get_state_for_audio_prompt'))

        # Check old methods don't exist
        self.assertFalse(hasattr(self.model, 'generate'))
        self.assertFalse(hasattr(self.model, 'stream'))
        self.assertFalse(hasattr(self.model, 'load_voice'))

    def test_parameter_names(self):
        """Test new parameter names work."""
        import inspect

        # Check load_model signature
        sig = inspect.signature(TTSModel.load_model)
        params = sig.parameters

        self.assertIn('lsd_decode_steps', params)
        self.assertIn('eos_threshold', params)
        self.assertNotIn('steps', params)
        self.assertNotIn('eos', params)

    def test_generation(self):
        """Test audio generation works."""
        audio = self.model.generate_audio(
            self.voice_state,
            "Test migration"
        )

        self.assertIsNotNone(audio)
        self.assertGreater(audio.shape[0], 0)

    def test_streaming(self):
        """Test streaming works."""
        chunks = list(self.model.generate_audio_stream(
            self.voice_state,
            "Test streaming"
        ))

        self.assertGreater(len(chunks), 0)

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

```python
# test_integration.py
import pytest
from pocket_tts import TTSModel

@pytest.fixture
def model():
    """Fixture for TTS model."""
    model = TTSModel.load_model()
    yield model
    # Cleanup
    del model

@pytest.fixture
def voice_state(model):
    """Fixture for voice state."""
    return model.get_state_for_audio_prompt("alba")

def test_end_to_end(model, voice_state):
    """Test complete generation pipeline."""
    # Generate audio
    audio = model.generate_audio(voice_state, "Integration test")

    # Verify output
    assert audio is not None
    assert audio.shape[0] > 0

    # Save to file
    import scipy.io.wavfile
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        output_path = Path(f.name)

    scipy.io.wavfile.write(
        str(output_path),
        model.sample_rate,
        audio.numpy()
    )

    # Verify file exists
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Cleanup
    output_path.unlink()
```

Run tests:
```bash
# Unit tests
python -m pytest test_migration.py -v

# Integration tests
python -m pytest test_integration.py -v

# All tests
python -m pytest tests/ -v
```

## Best Practices

### 1. Version Pinning

Always pin versions in production:

```txt
# requirements.txt
pocket-tts==1.0.0  # Exact version
# or
pocket-tts>=1.0.0,<1.1.0  # Version range
```

### 2. Compatibility Checking

Add version checks to your code:

```python
import pkg_resources

def check_pocket_tts_version():
    """Check Pocket TTS version."""
    required_version = "1.0.0"
    current_version = pkg_resources.get_distribution("pocket-tts").version

    if pkg_resources.parse_version(current_version) < pkg_resources.parse_version(required_version):
        raise RuntimeError(
            f"Pocket TTS {required_version}+ required, "
            f"but {current_version} is installed"
        )

check_pocket_tts_version()
```

### 3. Graceful Degradation

Handle version differences gracefully:

```python
try:
    # New API
    model = TTSModel.load_model()
except AttributeError:
    # Old API (for backward compatibility)
    model = TTSModel()
    print("Warning: Using deprecated API")
```

### 4. Migration Testing

Always test migrations in a staging environment before production:

```bash
# Development
export POCKET_TTS_ENV=dev
pip install 'pocket-tts>=1.0.0'

# Staging
export POCKET_TTS_ENV=staging
pip install 'pocket-tts>=1.0.0'

# Production (only after staging passes)
export POCKET_TTS_ENV=production
pip install 'pocket-tts>=1.0.0'
```

For more information, see:
- [Python API Documentation](python-api.md)
- [Configuration Guide](configuration-guide.md)
- [Release Notes](https://github.com/kyutai-labs/pocket-tts/releases)
