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

## Advanced Server Configuration

### Multi-Model Server

```python
from pocket_tts import TTSModel
import asyncio
import time

class MultiModelServer:
    """Server supporting multiple model variants."""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
    
    async def load_model(self, variant):
        """Load model variant on demand."""
        if variant not in self.models:
            print(f"Loading model variant: {variant}")
            self.models[variant] = TTSModel.load_model(variant=variant)
            self.models[variant].compile_for_inference()
        
        self.current_model = variant
        return self.models[variant]
    
    async def generate_with_variant(self, variant, voice, text):
        """Generate using specific model variant."""
        model = await self.load_model(variant)
        voice_state = model.get_state_for_audio_prompt(voice)
        return model.generate_audio(voice_state, text)

# FastAPI integration
@app.post("/generate/{variant}")
async def generate_with_variant(variant: str, request: GenerateRequest):
    try:
        server = MultiModelServer()
        audio = await server.generate_with_variant(variant, request.voice, request.text)
        return {"audio": audio.tolist(), "sample_rate": 24000}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Load Balancing Configuration

```bash
# Start multiple server instances
pocket-tts serve --port 8000 &
pocket-tts serve --port 8001 &
pocket-tts serve --port 8002 &

# Use nginx for load balancing
# nginx.conf
upstream tts_servers {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 80;
    
    location /tts/ {
        proxy_pass http://tts_servers;
        proxy_set_header Host $host;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    rustc \
    cargo \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV POCKET_TTS_DISABLE_GUI=1
ENV PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -m -u 1000 ttsuser
USER ttsuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from pocket_tts import TTSModel; TTSModel.load_model()" || exit 1

# Start server
CMD ["python", "main.py", "serve", "--host", "0.0.0.0"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  pocket-tts:
    build: .
    ports:
      - "8000:8000"
    environment:
      - POCKET_TTS_CACHE_DIR=/app/cache
      - POCKET_TTS_THREADS=4
    volumes:
      - ./cache:/app/cache
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Troubleshooting Server Issues

### Connection Timeout

```python
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import time

class RobustWebSocket:
    def __init__(self):
        self.active_connections = {}
        self.ping_interval = 30  # seconds
    
    async def handle_connection(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = {
            'websocket': websocket,
            'last_ping': time.time(),
            'generating': False
        }
        
        try:
            while True:
                try:
                    # Receive message with timeout
                    message = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=self.ping_interval
                    )
                    
                    data = json.loads(message)
                    await self.process_message(client_id, data)
                    
                except asyncio.TimeoutError:
                    # Send ping if no message received
                    if time.time() - self.active_connections[client_id]['last_ping'] > self.ping_interval * 2:
                        await websocket.send_json({"type": "ping"})
                        self.active_connections[client_id]['last_ping'] = time.time()
                        
        except WebSocketDisconnect:
            pass
        finally:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
```

### Memory Management

```python
import gc
import psutil

class MemoryEfficientServer:
    def __init__(self, max_memory_mb=2048):
        self.max_memory_mb = max_memory_mb
        self.request_count = 0
    
    async def generate_with_memory_check(self, voice, text):
        """Generate audio with memory monitoring."""
        # Check memory usage
        memory_mb = psutil.virtual_memory().used / 1024 / 1024
        
        if memory_mb > self.max_memory_mb:
            # Wait for memory to free up
            await asyncio.sleep(1)
            gc.collect()
            return await self.generate_with_memory_check(voice, text)
        
        try:
            voice_state = model.get_state_for_audio_prompt(voice)
            audio = model.generate_audio(voice_state, text)
            
            self.request_count += 1
            
            # Clean up every 10 requests
            if self.request_count % 10 == 0:
                gc.collect()
            
            return audio
            
        except Exception as e:
            # Force cleanup on error
            gc.collect()
            raise e
```

For more advanced usage, see the [Python API documentation](python-api.md) for direct integration with the TTS model, [Configuration Guide](configuration.md) for optimization options, and [Integration Examples](integration-examples.md) for complete application examples.
