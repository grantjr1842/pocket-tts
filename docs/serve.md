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

For more advanced usage, see the [Python API documentation](python-api.md) for direct integration with the TTS model.
