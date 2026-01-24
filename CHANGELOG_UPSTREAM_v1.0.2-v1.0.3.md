# Changelog: Upstream Changes v1.0.2 → v1.0.3

**Date:** January 15-22, 2026
**Source:** [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts)
**Total Commits:** 18

## Overview

This changelog documents all changes made to the upstream pocket-tts repository between versions 1.0.2 and 1.0.3. These changes include new features, bug fixes, documentation improvements, and community contributions.

---

## 🎉 New Features

### Voice Cloning with Multiple Audio Formats (#20)
**Commit:** `bedd1c4`

Voice cloning now supports multiple audio formats beyond just WAV:
- **Supported formats:** MP3, FLAC, OGG, M4A, and other formats compatible with `soundfile`
- **Implementation:** Replaced WAV-only `audio_read()` with `soundfile`-based implementation
- **Web UI:** Updated file picker to accept `.wav,.mp3,.flac,.ogg,.m4a,audio/*`
- **CLI:** Fixed missing `truncate=True` parameter in generate command

**Technical Changes:**
- Added `soundfile>=0.12.0` dependency to `pyproject.toml`
- Modified `pocket_tts/data/audio.py` to use `soundfile` library
- Updated `pocket_tts/main.py` to preserve original file extensions in temp files
- Added multi-channel audio support (automatically downmixes to mono)

### Chunk Size Control for Text Generation (#56)
**Commit:** `4ec89ac`

Added `max_tokens` parameter to the CLI `generate` command, allowing users to control the number of tokens per chunk when processing long texts.

**Benefits:**
- Users can now customize text chunking behavior
- Fixed tokenization issue by inserting commas for better sentence splitting
- Addresses issue #38

**Usage:**
```bash
pocket-tts generate "Your text here" --max-tokens 150
```

### KV Cache Memory Optimization (#32)
**Commit:** `8cadf71`

Implemented KV cache slicing when getting model state for voice cloning, significantly reducing memory usage for cached voice states.

**Technical Details:**
- Slices KV cache to only keep necessary frames for voice prompts
- Reduces memory footprint when caching voice states
- Improves efficiency for applications that reuse voice clones

---

## 🐛 Bug Fixes

### Respect `frames_after_eos` Parameter (#58)
**Commit:** `36f21a0`

Fixed the `generate_audio_stream()` and `generate_audio()` functions to properly respect the `frames_after_eos` parameter.

**Before:**
- Parameter was accepted but ignored
- Always used auto-calculated guess (3+ frames of silence)

**After:**
- User-specified `frames_after_eos` value is now respected
- Example: `frames_after_eos=0` produces minimal trailing silence
- Default behavior preserved when `frames_after_eos=None`

### Windows Temp File Fix (#72)
**Commit:** `2d62797`

Fixed temporary file handling issues on Windows by ensuring files are properly closed before being read back. This prevents file access errors on Windows systems.

---

## 📚 Documentation & Examples

### Colab Notebook Example (#67)
**Commit:** `6f9dd25`

Added an interactive Google Colab notebook demonstrating pocket-tts usage:
- Version-controlled notebook in the repository
- Easy browser-based experimentation
- Credits to @HarishDevLab for the suggestion

**File:** `docs/pocket-tts-example.ipynb`

### "Projects Using" Section (#45)
**Commit:** `f8f25ea`

Added a new section to the README showcasing community projects built with pocket-tts:
- Highlights real-world usage examples
- First featured project: [pocket-reader](https://github.com/lukasmwerner/pocket-reader) by @lukasmwerner
- Encourages community contributions

### Alternative Implementations Documentation

#### Rust Version (#55)
**Commit:** `54e9257`

Added documentation for the Rust implementation of pocket-tts in the alternative implementations section.
- Credits to @babybirdprd

#### Wyoming Satellite Project (#48)
**Commit:** `d35bf81`

Added link to [pocket-tts-wyoming](https://github.com/iankidd/pocket-tts-wyoming), a Wyoming satellite server integration.

#### Browser Implementations (#77)
**Commit:** `a3aaa13`

Added links to additional community implementations for running pocket-tts in browsers.

### Documentation Fixes
**Commits:** `630d519`, `83eac9f`

Various documentation improvements and README updates.

---

## 🔧 Infrastructure & CI/CD

### CI/CD Improvements

#### Do Not Use Auth in CI (#53)
**Commit:** `636ec23`

Removed authentication requirements from CI workflows, allowing anyone to run the CI.

#### Use Environment for PyPI Push (#54)
**Commit:** `8a2a29e`

Modified PyPI deployment to use GitHub Actions environments:
- Restricts PyPI publishing to users with approval rights
- Provides more flexible and secure CI/CD workflow
- Prevents unauthorized package publishing

---

## 🎨 Branding & Community

### Logo Update (#34)
**Commit:** `d1619d0`

Updated the project logo.

### Contribution Guidelines (#35)
**Commit:** `c4960e6`

Added rules and miscellaneous notes to contribution guidelines, improving clarity for potential contributors.

---

## 📦 Version Bumps

### Version 1.0.2 (#57)
**Commit:** `9650c07`

Version bump to 1.0.2 following the addition of multi-format audio support.

### Version 1.0.3 (#78)
**Commit:** `bfbaf80`

Version bump to 1.0.3 following additional bug fixes and features.

---

## 🤝 Community Contributions

This update included contributions from multiple community members:

- **@Und3rf10w** - Multi-format audio support for voice cloning
- **@Arihant Tripathy** - KV cache optimization and Windows temp file fixes
- **@BatteryShark** - `frames_after_eos` parameter fix
- **@Saleem Djima** - Chunk size control feature
- **@Lukas Werner** - "Projects using" section
- **@Ian Kidd** - Wyoming project link
- **@HarishDevLab** - Colab example suggestion
- **@babybirdprd** - Rust implementation

---

## 🔗 Related Issues

- Issue #14 - KV cache optimization
- Issue #20 - Multi-format audio support
- Issue #32 - KV cache slicing
- Issue #38 - Chunk size control
- Issue #39 - Projects using section
- Issue #45 - Community projects showcase
- Issue #48 - Wyoming integration
- Issue #53 - CI authentication
- Issue #54 - PyPI deployment environment
- Issue #55 - Rust implementation documentation
- Issue #56 - Max tokens parameter
- Issue #58 - frames_after_eos fix
- Issue #67 - Colab notebook
- Issue #72 - Windows temp file fix
- Issue #77 - Browser implementations

---

## 💡 Upgrade Notes

### For Users

1. **New Dependency:** `soundfile` is now required for voice cloning with non-WAV formats
   ```bash
   pip install soundfile
   # or
   pip install "pocket-tts[audio]"
   ```

2. **Voice Cloning:** You can now use MP3, FLAC, OGG, and other formats directly
   ```python
   # Previously: WAV only
   model_state = tts_model.get_state_for_audio_prompt("voice.wav")

   # Now: Any supported format
   model_state = tts_model.get_state_for_audio_prompt("voice.mp3")
   ```

3. **CLI Enhancement:** Control chunking behavior
   ```bash
   pocket-tts generate "long text..." --max-tokens 200
   ```

### For Developers

1. **KV Cache:** Voice state caching now uses less memory automatically
2. **API Changes:** `generate_audio_stream()` and `generate_audio()` now properly respect `frames_after_eos`
3. **File Upload:** When handling voice uploads, preserve the file extension for format detection

---

## 📊 Summary Statistics

- **Total changes:** 18 commits
- **New features:** 3 major features
- **Bug fixes:** 2 fixes
- **Documentation:** 5 improvements
- **Infrastructure:** 2 changes
- **Community showcases:** 3 additions
- **Contributors:** 8+ community members
- **Files modified:** 15+ files across the codebase

---

## 📝 Additional Resources

- **Original Repository:** [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts)
- **Demo:** [https://kyutai.org/pocket-tts](https://kyutai.org/pocket-tts)
- **Hugging Face:** [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)
- **Paper:** [arXiv:2509.06926](https://arxiv.org/abs/2509.06926)

---

*This changelog was generated on January 24, 2026, documenting changes merged from upstream into grantjr1842/pocket-tts fork.*
