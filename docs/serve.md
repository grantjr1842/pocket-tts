# Serve Command Documentation

The `serve` command starts a FastAPI web server that provides both a web interface and HTTP API for text-to-speech generation.

## Basic Usage

```bash
uvx pocket-tts serve
# or if installed manually:
pocket-tts serve
```

This starts a server on `http://localhost:8000` with the default voice model.

## Command Options

- `--voice VOICE`: Path to voice prompt audio file (voice to clone) (default: "hf://kyutai/tts-voices/alba-mackenna/casual.wav")
- `--host HOST`: Host to bind to (default: "localhost")
- `--port PORT`: Port to bind to (default: 8000)
- `--reload`: Enable auto-reload for development
- `--compile`: Enable `torch.compile` for inference (default: off)
- `--compile-backend`: torch.compile backend (default: "inductor")
- `--compile-mode`: torch.compile mode (default: "reduce-overhead")
- `--compile-fullgraph`: torch.compile fullgraph (default: false)
- `--compile-dynamic`: torch.compile dynamic (default: false)
- `--compile-targets`: Compile targets (all, flow-lm, mimi-decoder)

## Examples

### Basic Server

```bash
# Start with default settings
pocket-tts serve

# Custom host and port
pocket-tts serve --host "localhost" --port 8080
```

### Custom Voice

```bash
# Use different voice
pocket-tts serve --voice "hf://kyutai/tts-voices/jessica-jian/casual.wav"

# Use local voice file
pocket-tts serve --voice "./my_voice.wav"
```

## Web Interface

Once the server is running, navigate to `http://localhost:8000` to access the web interface.

## WebSocket Streaming API

The server also exposes a WebSocket endpoint for real-time streaming: `ws://<host>:<port>/ws/tts`.

### Protocol

- Client → server: `{"text": "Hello", "voice": "alba"}`
- Server → client: `{ "type": "audio", "data": "<base64>", "chunk": 0, "format": "wav" }`
- Subsequent chunks use `format: "pcm"` (16-bit PCM). The first chunk includes a WAV header.
- Server → client: `{ "type": "done", "total_chunks": N }`
- Heartbeat: server periodically sends `{ "type": "ping" }`, reply with `{ "type": "pong" }`.

### Browser example

```js
const ws = new WebSocket("ws://localhost:8000/ws/tts");
ws.addEventListener("open", () => {
  ws.send(JSON.stringify({ text: "Hello world", voice: "alba" }));
});

ws.addEventListener("message", (event) => {
  const payload = JSON.parse(event.data);
  if (payload.type === "ping") {
    ws.send(JSON.stringify({ type: "pong" }));
    return;
  }
  if (payload.type === "audio") {
    const bytes = Uint8Array.from(atob(payload.data), (c) => c.charCodeAt(0));
    // Feed bytes into your audio pipeline (WAV header in chunk 0, PCM thereafter).
  }
});
```

## Advanced Deployment

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install pocket-tts
RUN pip install --no-cache-dir pocket-tts

# Expose port
EXPOSE 8000

# Set environment variables
ENV POCKET_TTS_HOST=0.0.0.0
ENV POCKET_TTS_PORT=8000

# Run server
CMD ["pocket-tts", "serve"]
```

Build and run:

```bash
# Build image
docker build -t pocket-tts-server .

# Run container
docker run -p 8000:8000 pocket-tts-server

# Run with custom voice
docker run -p 8000:8000 \
  -v /path/to/voices:/voices:ro \
  -e POCKET_TTS_VOICE=/voices/custom.wav \
  pocket-tts-server
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  pocket-tts:
    build: .
    ports:
      - "8000:8000"
    environment:
      - POCKET_TTS_HOST=0.0.0.0
      - POCKET_TTS_PORT=8000
    volumes:
      - ./voices:/voices:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run:

```bash
docker-compose up -d
```

### Load Balancing

For high-traffic deployments, use a load balancer with multiple instances:

```yaml
# docker-compose.yml with multiple instances
version: '3.8'

services:
  pocket-tts-1:
    build: .
    ports:
      - "8001:8000"
    environment:
      - POCKET_TTS_PORT=8000

  pocket-tts-2:
    build: .
    ports:
      - "8002:8000"
    environment:
      - POCKET_TTS_PORT=8000

  nginx:
    image: nginx:alpine
    ports:
      - "8000:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - pocket-tts-1
      - pocket-tts-2
```

Nginx configuration (`nginx.conf`):

```nginx
events {
    worker_connections 1024;
}

http {
    upstream pocket_tts_backend {
        least_conn;
        server pocket-tts-1:8000;
        server pocket-tts-2:8000;
    }

    server {
        listen 80;

        location / {
            proxy_pass http://pocket_tts_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 300s;
        }
    }
}
```

### Production Server with Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn pocket_tts.serve:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

For more advanced usage, see the [Python API documentation](python-api.md) for direct integration with the TTS model.
