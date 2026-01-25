# Integration Examples

This guide provides comprehensive integration examples for Pocket TTS in various real-world scenarios and applications.

## Web Application Integration

### FastAPI Web Service

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import io
import tempfile
import asyncio
from pocket_tts import TTSModel, save_audio

app = FastAPI(title="Pocket TTS API")

class TTSRequest(BaseModel):
    text: str
    voice: str = "alba"
    temperature: float = 0.7
    format: str = "wav"

# Initialize model
model = TTSModel.load_model()
model.compile_for_inference()
voice_cache = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the TTS model on startup."""
    print("TTS API server starting up...")
    # Preload common voices
    common_voices = ["alba", "marius", "javert"]
    for voice in common_voices:
        voice_cache[voice] = model.get_state_for_audio_prompt(voice)
    print(f"Preloaded {len(common_voices)} voices")

@app.post("/generate")
async def generate_speech(request: TTSRequest):
    """Generate speech from text."""
    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long (max 5000 chars)")
        
        # Get voice state
        voice_state = voice_cache.get(request.voice)
        if not voice_state:
            voice_state = model.get_state_for_audio_prompt(request.voice)
            voice_cache[request.voice] = voice_state  # Cache for future
        
        # Generate audio
        audio = model.generate_audio(voice_state, request.text)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            save_audio(tmp_file.name, audio, model.sample_rate)
            return FileResponse(
                tmp_file.name,
                media_type="audio/wav",
                filename="generated_speech.wav"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-stream")
async def generate_stream(request: TTSRequest):
    """Generate streaming audio."""
    async def audio_streamer():
        try:
            voice_state = voice_cache.get(request.voice)
            if not voice_state:
                voice_state = model.get_state_for_audio_prompt(request.voice)
                voice_cache[request.voice] = voice_state
            
            # Stream audio chunks
            for chunk in model.generate_audio_stream(voice_state, request.text):
                yield chunk.numpy().tobytes()
                
        except Exception as e:
            yield f"Error: {str(e)}".encode()
    
    return StreamingResponse(
        audio_streamer(),
        media_type="application/octet-stream"
    )

@app.get("/voices")
async def list_voices():
    """List available voices."""
    return {
        "preset_voices": [
            "alba", "marius", "javert", "jean", 
            "fantine", "cosette", "eponine", "azelma"
        ],
        "cached_voices": list(voice_cache.keys())
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "cache_size": len(voice_cache),
        "device": model.device
    }
```

### WebSocket Real-time Service

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import asyncio
from pocket_tts import TTSModel

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass  # Connection might be closed

manager = ConnectionManager()

@app.websocket("/ws/tts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "generate":
                # Extract parameters
                text = message.get("text", "")
                voice = message.get("voice", "alba")
                temperature = message.get("temperature", 0.7)
                
                if not text.strip():
                    await manager.send_personal_message({
                        "type": "error",
                        "message": "Text cannot be empty"
                    }, websocket)
                    continue
                
                # Generate audio
                try:
                    voice_state = model.get_state_for_audio_prompt(voice)
                    
                    # Send chunks as they're generated
                    chunk_index = 0
                    for chunk in model.generate_audio_stream(voice_state, text):
                        await manager.send_personal_message({
                            "type": "audio_chunk",
                            "chunk": chunk_index,
                            "data": chunk.tolist(),
                            "sample_rate": model.sample_rate
                        }, websocket)
                        chunk_index += 1
                    
                    # Send completion signal
                    await manager.send_personal_message({
                        "type": "generation_complete",
                        "total_chunks": chunk_index
                    }, websocket)
                    
                except Exception as e:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": str(e)
                    }, websocket)
            
            elif message.get("type") == "ping":
                await manager.send_personal_message({"type": "pong"}, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

## Desktop Application Integration

### Tkinter GUI Application

```python
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
from pocket_tts import TTSModel, save_audio
import sounddevice as sd

class TTSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pocket TTS Desktop")
        self.root.geometry("800x600")
        
        # Initialize TTS model
        self.model = None
        self.voice_states = {}
        self.audio_queue = queue.Queue()
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """Create the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Voice selection
        ttk.Label(main_frame, text="Voice:").grid(row=0, column=0, sticky=tk.W)
        self.voice_var = tk.StringVar(value="alba")
        voice_combo = ttk.Combobox(main_frame, textvariable=self.voice_var)
        voice_combo['values'] = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
        voice_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Text input
        ttk.Label(main_frame, text="Text:").grid(row=1, column=0, sticky=tk.NW)
        self.text_widget = tk.Text(main_frame, height=10, wrap=tk.WORD)
        self.text_widget.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Generation controls
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.generate_btn = ttk.Button(control_frame, text="Generate", command=self.generate_audio)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = ttk.Button(control_frame, text="Play", command=self.play_audio, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(control_frame, text="Save", command=self.save_audio, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
    
    def load_model(self):
        """Load TTS model in background thread."""
        def load_worker():
            try:
                self.status_var.set("Loading model...")
                self.model = TTSModel.load_model()
                self.model.compile_for_inference()
                
                # Preload voices
                for voice in ["alba", "marius", "javert"]:
                    self.voice_states[voice] = self.model.get_state_for_audio_prompt(voice)
                
                self.status_var.set("Model loaded - Ready")
                self.generate_btn.config(state=tk.NORMAL)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
                self.status_var.set("Model loading failed")
        
        self.generate_btn.config(state=tk.DISABLED)
        threading.Thread(target=load_worker, daemon=True).start()
    
    def generate_audio(self):
        """Generate audio from text."""
        if not self.model:
            return
        
        text = self.text_widget.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter text to generate")
            return
        
        voice = self.voice_var.get()
        
        def generate_worker():
            try:
                self.status_var.set("Generating audio...")
                self.progress.start()
                
                voice_state = self.voice_states.get(voice)
                if not voice_state:
                    voice_state = self.model.get_state_for_audio_prompt(voice)
                    self.voice_states[voice] = voice_state
                
                self.current_audio = self.model.generate_audio(voice_state, text)
                
                self.progress.stop()
                self.status_var.set(f"Generated {len(self.current_audio)} samples")
                self.play_btn.config(state=tk.NORMAL)
                self.save_btn.config(state=tk.NORMAL)
                
            except Exception as e:
                self.progress.stop()
                messagebox.showerror("Error", f"Failed to generate audio: {e}")
                self.status_var.set("Generation failed")
        
        threading.Thread(target=generate_worker, daemon=True).start()
    
    def play_audio(self):
        """Play generated audio."""
        if hasattr(self, 'current_audio'):
            try:
                sd.play(self.current_audio.numpy(), self.model.sample_rate)
                self.status_var.set("Playing audio...")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to play audio: {e}")
    
    def save_audio(self):
        """Save generated audio to file."""
        if hasattr(self, 'current_audio'):
            filename = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
            )
            
            if filename:
                try:
                    save_audio(filename, self.current_audio, self.model.sample_rate)
                    self.status_var.set(f"Saved to {filename}")
                    messagebox.showinfo("Success", f"Audio saved to {filename}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save audio: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TTSApp(root)
    root.mainloop()
```

## Cloud Function Integration

### AWS Lambda Deployment

```python
import json
import base64
import tempfile
import os
from pocket_tts import TTSModel, save_audio

# Global model variable for reuse
model = None

def lambda_handler(event, context):
    """AWS Lambda handler for TTS generation."""
    global model
    
    try:
        # Initialize model if not already done
        if model is None:
            model = TTSModel.load_model()
            model.compile_for_inference()
            print("Model loaded in Lambda")
        
        # Parse request
        text = event.get('text', '')
        voice = event.get('voice', 'alba')
        output_format = event.get('format', 'base64')
        
        if not text.strip():
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Text cannot be empty'})
            }
        
        # Limit text length for Lambda
        if len(text) > 1000:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Text too long for Lambda (max 1000 chars)'})
            }
        
        # Generate audio
        voice_state = model.get_state_for_audio_prompt(voice)
        audio = model.generate_audio(voice_state, text)
        
        if output_format == 'base64':
            # Return as base64 encoded string
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                save_audio(tmp_file.name, audio, model.sample_rate)
                
                with open(tmp_file.name, 'rb') as f:
                    audio_bytes = f.read()
                
                os.unlink(tmp_file.name)
                
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json'
                    },
                    'body': json.dumps({
                        'audio': base64.b64encode(audio_bytes).decode(),
                        'sample_rate': model.sample_rate,
                        'format': 'wav'
                    })
                }
        
        else:
            # For other formats, upload to S3 and return URL
            import boto3
            s3 = boto3.client('s3')
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                save_audio(tmp_file.name, audio, model.sample_rate)
                
                filename = f"tts-{context.aws_request_id}.wav"
                s3.upload_file(tmp_file.name, os.environ['BUCKET_NAME'], filename)
                os.unlink(tmp_file.name)
                
                url = f"https://{os.environ['BUCKET_NAME']}.s3.amazonaws.com/{filename}"
                
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json'
                    },
                    'body': json.dumps({
                        'audio_url': url,
                        'sample_rate': model.sample_rate,
                        'format': 'wav'
                    })
                }
    
    except Exception as e:
        print(f"Lambda error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### Google Cloud Function Deployment

```python
import functions_framework
import base64
import tempfile
import os
from pocket_tts import TTSModel, save_audio

# Global model instance
model = None

@functions_framework.http
def generate_speech(request):
    """HTTP Cloud Function for TTS generation."""
    global model
    
    try:
        # Initialize model
        if model is None:
            model = TTSModel.load_model()
            model.compile_for_inference()
        
        # Parse request
        request_json = request.get_json()
        text = request_json.get('text', '')
        voice = request_json.get('voice', 'alba')
        
        if not text.strip():
            return ('Text cannot be empty', 400)
        
        # Generate audio
        voice_state = model.get_state_for_audio_prompt(voice)
        audio = model.generate_audio(voice_state, text)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            save_audio(tmp_file.name, audio, model.sample_rate)
            
            with open(tmp_file.name, 'rb') as f:
                audio_bytes = f.read()
            
            os.unlink(tmp_file.name)
        
        # Return as base64
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        response = {
            'audio': audio_b64,
            'sample_rate': model.sample_rate,
            'format': 'wav',
            'size_bytes': len(audio_bytes)
        }
        
        return (response, 200, {'Content-Type': 'application/json'})
    
    except Exception as e:
        return (f'Error: {str(e)}', 500)
```

## Chatbot Integration

### Discord Bot Integration

```python
import discord
import asyncio
import tempfile
import os
from pocket_tts import TTSModel, save_audio

class TTSBot(discord.Client):
    def __init__(self):
        super().__init__()
        self.model = None
        self.voice_states = {}
    
    async def on_ready(self):
        """Initialize model when bot is ready."""
        print(f'Logged in as {self.user}')
        await self.load_tts_model()
    
    async def load_tts_model(self):
        """Load TTS model in background."""
        await self.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="loading..."))
        
        # Load model in thread to avoid blocking
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(None, lambda: TTSModel.load_model())
        
        # Compile model
        await loop.run_in_executor(None, lambda: model.compile_for_inference())
        
        self.model = model
        
        # Preload voices
        voices = ["alba", "marius", "javert"]
        for voice in voices:
            voice_state = await loop.run_in_executor(
                None, 
                lambda v=voice: model.get_state_for_audio_prompt(v)
            )
            self.voice_states[voice] = voice_state
        
        await self.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="!tts"))
    
    async def on_message(self, message):
        """Handle incoming messages."""
        if message.author == self.user:
            return  # Don't respond to ourselves
        
        if message.content.startswith('!tts'):
            await self.handle_tts_command(message)
    
    async def handle_tts_command(self, message):
        """Handle TTS command."""
        try:
            # Parse command
            parts = message.content.split(maxsplit=2)
            
            if len(parts) < 2:
                await message.reply("Usage: `!tts [voice] text` or `!tts text`")
                return
            
            # Determine voice and text
            if len(parts) == 2:
                # !tts text (default voice)
                voice = "alba"
                text = parts[1]
            else:
                # !tts voice text
                voice = parts[1].lower()
                text = parts[2]
            
            # Validate voice
            valid_voices = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
            if voice not in valid_voices:
                await message.reply(f"Invalid voice. Valid voices: {', '.join(valid_voices)}")
                return
            
            # Generate audio
            await message.channel.typing()
            
            loop = asyncio.get_event_loop()
            
            # Get voice state
            voice_state = self.voice_states.get(voice)
            if not voice_state:
                voice_state = await loop.run_in_executor(
                    None, 
                    lambda: self.model.get_state_for_audio_prompt(voice)
                )
                self.voice_states[voice] = voice_state
            
            # Generate audio
            audio = await loop.run_in_executor(
                None,
                lambda: self.model.generate_audio(voice_state, text)
            )
            
            # Save to file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                await loop.run_in_executor(
                    None,
                    lambda: save_audio(tmp_file.name, audio, self.model.sample_rate)
                )
                
                # Send to voice channel if author is in one
                if message.author.voice:
                    voice_channel = message.author.voice.channel
                    voice_client = await voice_channel.connect()
                    
                    audio_source = discord.FFmpegPCMAudio(tmp_file.name)
                    voice_client.play(audio_source)
                    
                    # Wait for playback to finish
                    while voice_client.is_playing():
                        await asyncio.sleep(0.1)
                    
                    await voice_client.disconnect()
                else:
                    # Otherwise send as file
                    await message.reply("You're not in a voice channel, sending as file:", file=discord.File(tmp_file.name))
                
                os.unlink(tmp_file.name)
        
        except Exception as e:
            await message.reply(f"Error generating speech: {e}")

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = TTSBot(intents=intents)
bot.run('YOUR_BOT_TOKEN')
```

## Industrial Integration

### Command Line Tool Integration

```python
#!/usr/bin/env python3
import argparse
import sys
import json
from pathlib import Path
from pocket_tts import TTSModel, save_audio, load_wav

class TTSProcessor:
    """Command-line TTS processing tool."""
    
    def __init__(self):
        self.model = None
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file."""
        config_path = Path.home() / '.pocket-tts' / 'config.json'
        
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        
        return {
            "default_voice": "alba",
            "default_temperature": 0.7,
            "output_format": "wav",
            "sample_rate": 24000
        }
    
    def initialize_model(self, compile_model=False):
        """Initialize TTS model."""
        print("Loading TTS model...")
        self.model = TTSModel.load_model(
            temperature=self.config['default_temperature']
        )
        
        if compile_model:
            print("Compiling model for optimal performance...")
            self.model.compile_for_inference()
        
        print("Model ready.")
    
    def process_batch_file(self, input_file, output_dir, voice=None):
        """Process batch of texts from file."""
        input_path = Path(input_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not input_path.exists():
            print(f"Error: Input file {input_file} not found")
            return
        
        voice = voice or self.config['default_voice']
        voice_state = self.model.get_state_for_audio_prompt(voice)
        
        with open(input_path) as f:
            lines = f.readlines()
        
        print(f"Processing {len(lines)} lines...")
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                audio = self.model.generate_audio(voice_state, line)
                output_file = output_path / f"line_{i:04d}.wav"
                save_audio(output_file, audio, self.model.sample_rate)
                print(f"  Line {i}: {output_file}")
                
            except Exception as e:
                print(f"  Line {i} failed: {e}")
        
        print(f"Batch processing complete. Output in {output_path}")
    
    def process_single_text(self, text, output_file, voice=None):
        """Process single text input."""
        voice = voice or self.config['default_voice']
        voice_state = self.model.get_state_for_audio_prompt(voice)
        
        try:
            audio = self.model.generate_audio(voice_state, text)
            save_audio(output_file, audio, self.model.sample_rate)
            print(f"Audio saved to {output_file}")
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    def process_audio_to_voice(self, audio_file, output_name):
        """Convert audio file to voice state."""
        try:
            audio = load_wav(audio_file)
            voice_state = self.model.get_state_for_audio_prompt(audio)
            
            # Save voice state (conceptual - would need serialization)
            print(f"Voice state created from {audio_file}")
            print(f"Voice name: {output_name}")
            
        except Exception as e:
            print(f"Error processing audio: {e}")

def main():
    parser = argparse.ArgumentParser(description="Pocket TTS Command Line Tool")
    parser.add_argument("command", choices=["generate", "batch", "voice"], help="Command to execute")
    
    # Common arguments
    parser.add_argument("--voice", help="Voice to use")
    parser.add_argument("--output", help="Output file or directory")
    parser.add_argument("--compile", action="store_true", help="Compile model for performance")
    
    # Generate command arguments
    parser.add_argument("--text", help="Text to generate")
    
    # Batch command arguments
    parser.add_argument("--input", help="Input file for batch processing")
    
    args = parser.parse_args()
    
    processor = TTSProcessor()
    processor.initialize_model(args.compile)
    
    if args.command == "generate":
        if not args.text:
            text = sys.stdin.read()
        else:
            text = args.text
        
        if not args.output:
            args.output = "output.wav"
        
        processor.process_single_text(text, args.output, args.voice)
    
    elif args.command == "batch":
        if not args.input:
            print("Error: --input required for batch processing")
            sys.exit(1)
        
        if not args.output:
            args.output = "./batch_output"
        
        processor.process_batch_file(args.input, args.output, args.voice)
    
    elif args.command == "voice":
        if not args.input:
            print("Error: --input required for voice extraction")
            sys.exit(1)
        
        output_name = args.output or "custom_voice"
        processor.process_audio_to_voice(args.input, output_name)

if __name__ == "__main__":
    main()
```

These integration examples show how to effectively integrate Pocket TTS into various types of applications and deployment scenarios.