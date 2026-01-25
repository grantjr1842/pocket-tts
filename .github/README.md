# GitHub Actions CI/CD

This directory contains the GitHub Actions workflows for continuous integration and testing of the Pocket TTS project.

## Workflows

### 1. Tests (`.github/workflows/test.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

**Matrix:**
- **OS:** Ubuntu, macOS, Windows
- **Python:** 3.10, 3.11, 3.12, 3.13, 3.14

**Steps:**
1. Set up Python environment
2. Install dependencies with `uv`
3. Run linting (`ruff check` and `ruff format --check`)
4. Run tests with coverage (`pytest --cov=pocket_tts`)
5. Upload coverage to Codecov (Ubuntu + Python 3.12 only)

### 2. Integration Tests (`.github/workflows/integration.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

**Tests:**
- CLI functionality smoke test
- Model loading smoke test
- Audio I/O functionality test
- NumPy replacement functionality test
- Rust extension building and integration test

### 3. Performance Tests (`.github/workflows/performance.yml`)

**Triggers:**
- Push to `main` branch
- Pull requests to `main` branch
- Daily schedule (2 AM UTC)

**Tests:**
- Model loading time benchmark
- Voice state creation time benchmark
- Audio generation speed benchmark
- Memory usage analysis
- Rust vs NumPy performance comparison

### 4. Security and Dependency Checks (`.github/workflows/security.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Weekly schedule (3 AM UTC on Sundays)

**Checks:**
- Security vulnerability scan (`pip-audit`)
- Rust dependency audit (`cargo audit`)
- Dependency tree analysis
- License compatibility check
- Code quality checks
- Anti-pattern detection

## Development Guidelines

### Running Tests Locally

Before pushing, make sure tests pass locally:

```bash
# Install dependencies
uv sync --dev

# Run linting
uv run ruff check .
uv run ruff format .

# Run tests with coverage
uv run pytest tests/ -v --cov=pocket_tts --cov-report=term-missing

# Run integration tests
uv run pytest tests/ -v -m integration
```

### Adding New Tests

1. Unit tests go in `tests/` directory
2. Integration tests should be marked with `@pytest.mark.integration`
3. Performance tests should be added to the performance workflow
4. Update workflow files if new test categories are added

### Debugging CI Failures

1. Check the workflow logs on GitHub Actions
2. Replicate the environment locally:
   ```bash
   uv sync --dev
   uv run pytest tests/ -v -k "failing_test_name"
   ```
3. For OS-specific issues, test in a similar environment using Docker or GitHub Codespaces

### Performance Monitoring

- Performance benchmarks are tracked over time
- Significant regressions will be flagged in PRs
- Daily runs provide baseline performance data

### Security

- Security scans run automatically on all PRs
- Vulnerabilities are reported but don't block builds by default
- Critical security issues should be addressed promptly

## Configuration

### Environment Variables

The workflows use the following environment variables (automatically provided by GitHub Actions):

- `GITHUB_TOKEN`: For API access
- `PYTHON_VERSION`: Matrix strategy variable
- `OS`: Matrix strategy variable

### Secrets

- `CODECOV_TOKEN`: For uploading coverage reports (configured in repository settings)

## Troubleshooting

### Common Issues

1. **Rust Extension Build Failures**
   - Check that Rust toolchain is properly installed
   - Verify `training/rust_exts/audio_ds/Cargo.toml` exists
   - Ensure all Rust dependencies are available

2. **Test Timeouts**
   - Performance tests may timeout on slow runners
   - Model loading tests require sufficient memory
   - Consider reducing test data size for CI

3. **Coverage Upload Failures**
   - Verify Codecov token is configured
   - Check that coverage XML file is generated
   - Ensure coverage file paths are correct

### Getting Help

- Check GitHub Actions documentation
- Review workflow logs for detailed error messages
- Open an issue for persistent CI problems
