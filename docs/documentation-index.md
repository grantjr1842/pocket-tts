# Pocket TTS Documentation Index

Complete navigation guide for all Pocket TTS documentation.

## Quick Start

- [README](../README.md) - Project overview, installation, and quick start guide
- [Python API Documentation](python-api.md) - Get started with the Python API
- [Generate Command Documentation](generate.md) - Learn the CLI generate command
- [Serve Command Documentation](serve.md) - Learn the CLI serve command

## Core Documentation

### Getting Started
- [README](../README.md) - Project overview, installation, and basic usage
- [Python API Documentation](python-api.md) - Complete Python API reference and examples
- [Generate Command Documentation](generate.md) - CLI reference for audio generation
- [Serve Command Documentation](serve.md) - CLI reference for web server

### Configuration
- [Configuration Guide](configuration-guide.md) - Comprehensive configuration options
  - Environment variables
  - Model configuration parameters
  - Compilation settings
  - Voice configuration
  - Server configuration
  - Performance tuning

### Development
- [Command Map](command-map.md) - Development commands (format, lint, test, build)
- [Error Handling Guide](error-handling-guide.md) - Robust error handling patterns
- [Performance Optimization Guide](performance-optimization-guide.md) - Optimization techniques

## Advanced Topics

### Integration
- [Integration Examples](integration-examples.md) - Complete integration examples
  - FastAPI web service
  - Tkinter desktop application
  - AWS Lambda function
  - Google Cloud function
  - Discord bot
  - Command-line tool
  - Flask web service
  - AsyncIO integration

### Operations
- [Migration Guide](migration-guide.md) - Version compatibility and migration
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions

## Reference

### API Reference
- [Python API Documentation](python-api.md) - Complete API reference
- [Configuration Guide](configuration-guide.md) - All configuration options

### CLI Reference
- [Generate Command Documentation](generate.md) - Generate command reference
- [Serve Command Documentation](serve.md) - Serve command reference

### Guides
- [Error Handling Guide](error-handling-guide.md) - Error handling patterns
- [Performance Optimization Guide](performance-optimization-guide.md) - Performance techniques
- [Migration Guide](migration-guide.md) - Version migration procedures

## Documentation by Use Case

### For Users

#### Quick Start
1. Read the [README](../README.md) for overview
2. Try the [Generate Command](generate.md) for CLI usage
3. Try the [Serve Command](serve.md) for web interface

#### Basic Python Usage
1. Read [Python API Documentation - Quick Start](python-api.md#quick-start)
2. Review [Core Classes](python-api.md#core-classes)
3. Explore [Advanced Usage](python-api.md#advanced-usage)

#### Configuration
1. Read [Configuration Guide](configuration-guide.md)
2. Set up [Environment Variables](configuration-guide.md#environment-variables)
3. Configure [Model Parameters](configuration-guide.md#model-configuration)

### For Developers

#### Integration
1. Read [Integration Examples](integration-examples.md)
2. Choose your integration type:
   - [FastAPI Web Service](integration-examples.md#fastapi-web-service)
   - [Tkinter Desktop Application](integration-examples.md#tkinter-desktop-application)
   - [AWS Lambda Function](integration-examples.md#aws-lambda-function)
   - [Google Cloud Function](integration-examples.md#google-cloud-function)
   - [Discord Bot](integration-examples.md#discord-bot)

#### Error Handling
1. Read [Error Handling Guide](error-handling-guide.md)
2. Implement [Error Handling Patterns](error-handling-guide.md#error-handling-patterns)
3. Add [Retry Strategies](error-handling-guide.md#retry-strategies)

#### Performance
1. Read [Performance Optimization Guide](performance-optimization-guide.md)
2. Apply [Model Compilation](performance-optimization-guide.md#model-compilation)
3. Implement [Memory Management](performance-optimization-guide.md#memory-management)
4. Monitor with [System Resource Monitoring](performance-optimization-guide.md#system-resource-monitoring)

#### Development Workflow
1. Follow [Command Map](command-map.md) for development commands
2. Run [Format](command-map.md#format) and [Lint](command-map.md#lint)
3. Execute [Tests](command-map.md#tests)
4. Create [Build](command-map.md#build)

### For DevOps

#### Deployment
1. Read [Configuration Guide - Server Configuration](configuration-guide.md#server-configuration)
2. Follow [Docker Deployment](serve.md#docker-deployment)
3. Set up [Load Balancing](serve.md#load-balancing)

#### Operations
1. Review [Troubleshooting Guide](troubleshooting.md)
2. Implement [Performance Monitoring](performance-optimization-guide.md#system-resource-monitoring)
3. Plan [Migrations](migration-guide.md) when upgrading

## Documentation Structure

```
docs/
├── README.md (This file - documentation index)
├── configuration-guide.md
├── documentation-index.md (This file)
├── error-handling-guide.md
├── integration-examples.md
├── migration-guide.md
├── performance-optimization-guide.md
├── command-map.md
├── generate.md
├── python-api.md
├── serve.md
└── troubleshooting.md
```

## Common Tasks

### Generate Audio from Text
- **CLI**: [Generate Command Documentation](generate.md)
- **Python**: [Python API Documentation](python-api.md)

### Deploy a TTS Server
- **Docker**: [Docker Deployment](serve.md#docker-deployment)
- **Load Balancing**: [Load Balancing](serve.md#load-balancing)
- **Configuration**: [Server Configuration](configuration-guide.md#server-configuration)

### Optimize Performance
- **Compilation**: [Model Compilation](performance-optimization-guide.md#model-compilation)
- **Memory**: [Memory Management](performance-optimization-guide.md#memory-management)
- **CPU**: [CPU Optimization](performance-optimization-guide.md#cpu-optimization)

### Handle Errors
- **Patterns**: [Error Handling Patterns](error-handling-guide.md#error-handling-patterns)
- **Retries**: [Retry Strategies](error-handling-guide.md#retry-strategies)
- **Troubleshooting**: [Troubleshooting Guide](troubleshooting.md)

### Integrate into Applications
- **FastAPI**: [FastAPI Web Service](integration-examples.md#fastapi-web-service)
- **Desktop**: [Tkinter Desktop Application](integration-examples.md#tkinter-desktop-application)
- **Serverless**: [AWS Lambda](integration-examples.md#aws-lambda-function) | [Google Cloud](integration-examples.md#google-cloud-function)
- **Bots**: [Discord Bot](integration-examples.md#discord-bot)

### Upgrade Versions
- **Migration**: [Migration Guide](migration-guide.md)
- **Breaking Changes**: [Breaking Changes](migration-guide.md#breaking-changes)
- **Rollback**: [Rollback Procedures](migration-guide.md#rollback-procedures)

## Additional Resources

- [GitHub Repository](https://github.com/kyutai-labs/pocket-tts) - Source code and issues
- [Hugging Face Model Card](https://huggingface.co/kyutai/pocket-tts) - Model information
- [Tech Report](https://kyutai.org/blog/2026-01-13-pocket-tts) - Technical details
- [Paper](https://arxiv.org/abs/2509.06926) - Research paper
- [Demo](https://kyutai.org/pocket-tts) - Live demo

## Getting Help

If you can't find what you're looking for:

1. **Search the documentation**: Use your browser's find function (Ctrl+F / Cmd+F)
2. **Check the Troubleshooting Guide**: [Troubleshooting Guide](troubleshooting.md)
3. **Search GitHub Issues**: [Pocket TTS Issues](https://github.com/kyutai-labs/pocket-tts/issues)
4. **Create a new issue**: Use the GitHub issue tracker with:
   - Clear description of the problem
   - Steps to reproduce
   - System information (OS, Python version, etc.)
   - Error messages and tracebacks

## Contributing to Documentation

To improve the documentation:

1. Edit the relevant markdown file in the `docs/` directory
2. Follow the existing formatting and style
3. Test code examples before submitting
4. Update links if adding or moving files
5. Submit a pull request with your changes

For more information, see the [CONTRIBUTING.md](../CONTRIBUTING.md) file.
