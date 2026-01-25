# Pocket TTS Integration Examples

This guide provides complete, working examples for integrating Pocket TTS into various applications and platforms.

## Table of Contents

- [FastAPI Web Service](#fastapi-web-service)
- [Tkinter Desktop Application](#tkinter-desktop-application)
- [AWS Lambda Function](#aws-lambda-function)
- [Google Cloud Function](#google-cloud-function)
- [Discord Bot](#discord-bot)
- [Command-Line Tool](#command-line-tool)
- [Flask Web Service](#flask-web-service)
- [AsyncIO Integration](#asyncio-integration)

## FastAPI Web Service

Complete FastAPI service with async support, error handling, and file management.

### Implementation

```python
# tts_service.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
from typing import Optional
import logging
import uuid
from pathlib import Path
import torch
import scipy.io.wavfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pocket TTS API",
    description="Text-to-Speech API using Pocket TTS",
    version="1.0.0"
)

# Global TTS model and voice cache
class TTSManager:
    def __init__(self):
        self.model = None
        self.voices = {}
        self.initialized = False

    def initialize(self):
        if not self.initialized:
            logger.info("Initializing TTS model...")
            from pocket_tts import TTSModel
            self.model = TTSModel.load_model()
            self.model.compile_for_inference(mode="reduce-overhead")
            self.initialized = True
            logger.info("TTS model initialized")

    def get_voice(self, voice_name: str):
        if voice_name not in self.voices:
            logger.info(f"Loading voice: {voice_name}")
            self.voices[voice_name] = self.model.get_state_for_audio_prompt(voice_name)
        return self.voices[voice_name]

tts_manager = TTSManager()

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    tts_manager.initialize()

# Request models
class GenerationRequest(BaseModel):
    text: str
    voice: str = "alba"
    temperature: Optional[float] = None
    decode_steps: Optional[int] = None

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        if len(v) > 5000:
            raise ValueError("Text too long (max 5000 characters)")
        return v

class GenerationResponse(BaseModel):
    status: str
    file_id: str
    download_url: str
    duration_seconds: float

# Storage for generated files
generated_files = {}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Pocket TTS API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/generate",
            "download": "/download/{file_id}",
            "health": "/health",
            "voices": "/voices"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": tts_manager.initialized,
        "cached_voices": len(tts_manager.voices)
    }

@app.get("/voices")
async def list_voices():
    """List available voices."""
    predefined = [
        "alba", "marius", "javert", "jean",
        "fantine", "cosette", "eponine", "azelma"
    ]
    return {
        "predefined_voices": predefined,
        "cached_voices": list(tts_manager.voices.keys())
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_audio(
    request: GenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate audio from text."""
    file_id = str(uuid.uuid4())
    output_path = Path(f"/tmp/tts_{file_id}.wav")

    try:
        logger.info(f"Generating audio: voice={request.voice}, text_length={len(request.text)}")

        # Get voice state
        voice_state = tts_manager.get_voice(request.voice)

        # Generate audio
        audio = tts_manager.model.generate_audio(voice_state, request.text)

        # Save to file
        scipy.io.wavfile.write(
            str(output_path),
            tts_manager.model.sample_rate,
            audio.numpy()
        )

        # Calculate duration
        duration = audio.shape[0] / tts_manager.model.sample_rate

        # Store file for cleanup
        generated_files[file_id] = output_path

        # Schedule cleanup after 1 hour
        background_tasks.add_task(cleanup_file, file_id, delay=3600)

        logger.info(f"Audio generated: file_id={file_id}, duration={duration:.2f}s")

        return GenerationResponse(
            status="success",
            file_id=file_id,
            download_url=f"/download/{file_id}",
            duration_seconds=duration
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{file_id}")
async def download_audio(file_id: str):
    """Download generated audio file."""
    if file_id not in generated_files:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = generated_files[file_id]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=f"tts_{file_id}.wav"
    )

async def cleanup_file(file_id: str, delay: int = 0):
    """Clean up generated file after delay."""
    import asyncio
    if delay > 0:
        await asyncio.sleep(delay)

    if file_id in generated_files:
        file_path = generated_files[file_id]
        try:
            file_path.unlink()
            del generated_files[file_id]
            logger.info(f"Cleaned up file: {file_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup file {file_id}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Running the Service

```bash
# Install dependencies
pip install fastapi uvicorn pocket-tts

# Run the service
python tts_service.py

# Or with uvicorn directly
uvicorn tts_service:app --host 0.0.0.0 --port 8000 --reload
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# List voices
curl http://localhost:8000/voices

# Generate audio
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "voice": "alba"}'

# Download audio
curl http://localhost:8000/download/{file_id} --output output.wav
```

## Tkinter Desktop Application

Complete desktop application with GUI for TTS generation.

### Implementation

```python
# tts_gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
import scipy.io.wavfile
from typing import Optional

class TTSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pocket TTS")
        self.root.geometry("600x500")

        # TTS model (loaded in background)
        self.model = None
        self.voices = {}
        self.is_generating = False

        # Create GUI
        self.create_widgets()

        # Initialize model in background
        threading.Thread(target=self.initialize_model, daemon=True).start()

    def create_widgets(self):
        """Create GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Status label
        self.status_label = ttk.Label(
            main_frame,
            text="Initializing model...",
            foreground="blue"
        )
        self.status_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Voice selection
        ttk.Label(main_frame, text="Voice:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.voice_var = tk.StringVar(value="alba")
        voice_combo = ttk.Combobox(
            main_frame,
            textvariable=self.voice_var,
            values=["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"],
            state="readonly",
            width=40
        )
        voice_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)

        # Text input
        ttk.Label(main_frame, text="Text:").grid(row=2, column=0, sticky=tk.NW, pady=5)
        self.text_input = tk.Text(main_frame, height=10, width=50, wrap=tk.WORD)
        self.text_input.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        self.text_input.insert("1.0", "Hello world! This is Pocket TTS.")
        self.text_input.configure(state='normal')

        # Output path
        ttk.Label(main_frame, text="Output:").grid(row=3, column=0, sticky=tk.W, pady=5)
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)

        self.output_path_var = tk.StringVar(value="output.wav")
        output_entry = ttk.Entry(output_frame, textvariable=self.output_path_var, width=35)
        output_entry.pack(side=tk.LEFT, padx=(0, 5))

        browse_button = ttk.Button(
            output_frame,
            text="Browse...",
            command=self.browse_output
        )
        browse_button.pack(side=tk.LEFT)

        # Progress bar
        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            length=400
        )
        self.progress.grid(row=4, column=0, columnspan=2, pady=10)

        # Generate button
        self.generate_button = ttk.Button(
            main_frame,
            text="Generate Audio",
            command=self.generate_audio,
            state=tk.DISABLED
        )
        self.generate_button.grid(row=5, column=0, columnspan=2, pady=10)

        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def initialize_model(self):
        """Initialize TTS model in background thread."""
        try:
            from pocket_tts import TTSModel
            self.model = TTSModel.load_model()
            self.model.compile_for_inference(mode="reduce-overhead")
            self.update_status("Model loaded successfully", "green")
            self.generate_button.configure(state=tk.NORMAL)
        except Exception as e:
            self.update_status(f"Failed to load model: {e}", "red")
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def update_status(self, message: str, color: str = "blue"):
        """Update status label."""
        self.status_label.configure(text=message, foreground=color)

    def browse_output(self):
        """Browse for output file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if filename:
            self.output_path_var.set(filename)

    def get_voice_state(self, voice: str):
        """Get or create voice state."""
        if voice not in self.voices:
            self.voices[voice] = self.model.get_state_for_audio_prompt(voice)
        return self.voices[voice]

    def generate_audio(self):
        """Generate audio in background thread."""
        if self.is_generating:
            return

        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter text to generate")
            return

        voice = self.voice_var.get()
        output_path = self.output_path_var.get()

        # Disable button and show progress
        self.is_generating = True
        self.generate_button.configure(state=tk.DISABLED)
        self.progress.start()
        self.update_status("Generating audio...", "blue")

        # Generate in background
        thread = threading.Thread(
            target=self._generate_audio_thread,
            args=(text, voice, output_path),
            daemon=True
        )
        thread.start()

    def _generate_audio_thread(self, text: str, voice: str, output_path: str):
        """Background thread for audio generation."""
        try:
            # Get voice state
            voice_state = self.get_voice_state(voice)

            # Generate audio
            audio = self.model.generate_audio(voice_state, text)

            # Save to file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            scipy.io.wavfile.write(
                str(output_path),
                self.model.sample_rate,
                audio.numpy()
            )

            # Update GUI (must be done in main thread)
            self.root.after(0, self._generation_complete, True, None)

        except Exception as e:
            self.root.after(0, self._generation_complete, False, str(e))

    def _generation_complete(self, success: bool, error: Optional[str]):
        """Handle generation completion."""
        self.progress.stop()
        self.is_generating = False
        self.generate_button.configure(state=tk.NORMAL)

        if success:
            self.update_status("Audio generated successfully!", "green")
            messagebox.showinfo(
                "Success",
                f"Audio saved to {self.output_path_var.get()}"
            )
        else:
            self.update_status("Generation failed", "red")
            messagebox.showerror("Error", f"Failed to generate audio: {error}")

def main():
    root = tk.Tk()
    app = TTSApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
```

### Running the Application

```bash
# Install dependencies
pip install pocket-ttk

# Run the application
python tts_gui.py
```

## AWS Lambda Function

Serverless TTS function for AWS Lambda.

### Implementation

```python
# lambda_function.py
import json
import os
import tempfile
from pathlib import Path
import boto3
import scipy.io.wavfile
from pocket_tts import TTSModel

# Initialize model outside handler for reuse
MODEL = None
VOICES = {}

def get_model():
    """Get or initialize TTS model."""
    global MODEL
    if MODEL is None:
        print("Initializing TTS model...")
        MODEL = TTSModel.load_model()
        MODEL.compile_for_inference(mode="reduce-overhead")
        print("Model initialized")
    return MODEL

def get_voice_state(voice: str):
    """Get or create voice state."""
    if voice not in VOICES:
        VOICES[voice] = get_model().get_state_for_audio_prompt(voice)
    return VOICES[voice]

def lambda_handler(event, context):
    """AWS Lambda handler."""
    try:
        # Parse request
        body = event.get('body', '{}')
        if isinstance(body, str):
            body = json.loads(body)

        text = body.get('text', '')
        voice = body.get('voice', 'alba')
        response_type = body.get('response_type', 'base64')  # 'base64' or 's3'

        # Validate input
        if not text:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Text is required'})
            }

        # Generate audio
        model = get_model()
        voice_state = get_voice_state(voice)
        audio = model.generate_audio(voice_state, text)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            scipy.io.wavfile.write(
                temp_path,
                model.sample_rate,
                audio.numpy()
            )

        # Return response
        if response_type == 's3':
            # Upload to S3
            s3_key = f"tts/{context.request_id}.wav"
            s3_client = boto3.client('s3')
            bucket_name = os.environ.get('S3_BUCKET_NAME')

            if not bucket_name:
                return {
                    'statusCode': 500,
                    'body': json.dumps({'error': 'S3_BUCKET_NAME not configured'})
                }

            s3_client.upload_file(
                temp_path,
                bucket_name,
                s3_key,
                ExtraArgs={'ContentType': 'audio/wav'}
            )

            # Generate presigned URL
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': s3_key},
                ExpiresIn=3600
            )

            response = {
                'statusCode': 200,
                'body': json.dumps({
                    'status': 'success',
                    'url': url,
                    'duration': audio.shape[0] / model.sample_rate
                })
            }

        else:  # base64
            # Encode as base64
            import base64
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')

            response = {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'audio/wav',
                    'Content-Disposition': 'attachment; filename="speech.wav"'
                },
                'body': audio_base64,
                'isBase64Encoded': True
            }

        # Cleanup temp file
        Path(temp_path).unlink(missing_ok=True)

        return response

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### AWS SAM Template

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Parameters:
  S3BucketName:
    Type: String
    Description: S3 bucket for storing generated audio

Globals:
  Function:
    Timeout: 30
    MemorySize: 1024
    Environment:
      Variables:
        S3_BUCKET_NAME: !Ref S3BucketName

Resources:
  TTSFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: ./lambda_function.py
      Handler: lambda_function.lambda_handler
      Runtime: python3.11
      Layers:
        - !Ref PocketTTSLayer
      Events:
        GenerateAPI:
          Type: Api
          Properties:
            Path: /generate
            Method: post

  PocketTTSLayer:
    Type: AWS::Serverless::LayerVersion
    Properties:
      ContentUri: ./layer/
      CompatibleRuntimes:
        - python3.11

Outputs:
  ApiURL:
    Description: API endpoint URL
    Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/generate"
```

### Deploying to AWS Lambda

```bash
# Install AWS SAM CLI
pip install aws-sam-cli

# Build and deploy
sam build
sam deploy --guided

# Test the function
sam local invoke "TTSFunction" -e event.json
```

## Google Cloud Function

Serverless TTS function for Google Cloud.

### Implementation

```python
# main.py
import os
import tempfile
from pathlib import Path
from flask import escape
import scipy.io.wavfile
from pocket_tts import TTSModel
from google.cloud import storage

# Initialize model
MODEL = None
VOICES = {}

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = TTSModel.load_model()
        MODEL.compile_for_inference(mode="reduce-overhead")
    return MODEL

def get_voice_state(voice: str):
    if voice not in VOICES:
        VOICES[voice] = get_model().get_state_for_audio_prompt(voice)
    return VOICES[voice]

def generate_tts(request):
    """HTTP Cloud Function."""
    # CORS headers
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {'Access-Control-Allow-Origin': '*'}

    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return ('Invalid JSON', 400, headers)

        text = request_json.get('text', '')
        voice = request_json.get('voice', 'alba')
        response_type = request_json.get('response_type', 'base64')

        # Validate
        if not text:
            return ('Text is required', 400, headers)

        # Generate audio
        model = get_model()
        voice_state = get_voice_state(voice)
        audio = model.generate_audio(voice_state, text)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            scipy.io.wavfile.write(
                temp_path,
                model.sample_rate,
                audio.numpy()
            )

        if response_type == 'gcs':
            # Upload to Google Cloud Storage
            bucket_name = os.environ.get('GCS_BUCKET_NAME')
            if not bucket_name:
                return ('GCS_BUCKET_NAME not configured', 500, headers)

            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob_name = f"tts/{os.urandom(16).hex()}.wav"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(temp_path, content_type='audio/wav')

            # Generate signed URL
            url = blob.generate_signed_url(expiration=3600)

            response_data = {
                'status': 'success',
                'url': url,
                'duration': audio.shape[0] / model.sample_rate
            }
            response = (json.dumps(response_data), 200, headers)

        else:  # base64
            import base64
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')

            response = (audio_base64, 200, {**headers, 'Content-Type': 'audio/wav'})

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
        return response

    except Exception as e:
        return (f'Error: {str(e)}', 500, headers)
```

### Deploying to Google Cloud

```bash
# Deploy the function
gcloud functions deploy pocket-tts \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --memory 1GB \
  --timeout 30s \
  --set-env-vars GCS_BUCKET_NAME=your-bucket

# Test the function
curl -X POST https://YOUR_REGION-YOUR_PROJECT_ID.cloudfunctions.net/pocket-tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "voice": "alba"}'
```

## Discord Bot

Discord bot that generates TTS audio for text messages.

### Implementation

```python
# discord_bot.py
import discord
from discord.ext import commands
import tempfile
from pathlib import Path
import scipy.io.wavfile
from pocket_tts import TTSModel
import os

# Bot configuration
TOKEN = os.environ.get('DISCORD_TOKEN')
PREFIX = '!'

# Initialize bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# TTS model (initialize when ready)
tts_model = None
voice_states = {}

@bot.event
async def on_ready():
    """Initialize TTS model when bot is ready."""
    global tts_model
    print(f'{bot.user} has connected to Discord!')
    print('Initializing TTS model...')
    tts_model = TTSModel.load_model()
    tts_model.compile_for_inference(mode="reduce-overhead")
    print('TTS model ready!')

def get_voice_state(voice: str):
    """Get or create voice state."""
    if voice not in voice_states:
        voice_states[voice] = tts_model.get_state_for_audio_prompt(voice)
    return voice_states[voice]

@bot.command(name='tts')
async def tts_command(ctx, *, text: str):
    """Generate TTS audio from text."""
    if len(text) > 500:
        await ctx.send("Text too long! Maximum 500 characters.")
        return

    # Send typing indicator
    async with ctx.typing():
        try:
            # Generate audio
            voice_state = get_voice_state("alba")
            audio = tts_model.generate_audio(voice_state, text)

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
                scipy.io.wavfile.write(
                    temp_path,
                    tts_model.sample_rate,
                    audio.numpy()
                )

            # Send audio file
            file = discord.File(temp_path, filename='tts.wav')
            await ctx.send(file=file)

            # Cleanup
            Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            await ctx.send(f"Error generating audio: {str(e)}")

@bot.command(name='voice')
async def voice_command(ctx, voice: str):
    """Set the voice for TTS."""
    valid_voices = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
    if voice not in valid_voices:
        await ctx.send(f"Invalid voice. Valid voices: {', '.join(valid_voices)}")
        return

    # Preload voice
    try:
        get_voice_state(voice)
        await ctx.send(f"Voice set to: {voice}")
    except Exception as e:
        await ctx.send(f"Error loading voice: {str(e)}")

@bot.command(name='voices')
async def voices_command(ctx):
    """List available voices."""
    voices = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
    await ctx.send(f"Available voices: {', '.join(voices)}")

@bot.event
async def on_command_error(ctx, error):
    """Handle command errors."""
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Missing required argument.")
    elif isinstance(error, commands.CommandNotFound):
        pass  # Ignore unknown commands
    else:
        print(f"Error: {error}")

if __name__ == "__main__":
    if not TOKEN:
        print("Error: DISCORD_TOKEN environment variable not set")
    else:
        bot.run(TOKEN)
```

### Requirements

```txt
# requirements.txt
discord.py
pocket-tts
scipy
```

### Running the Bot

```bash
# Install dependencies
pip install -r requirements.txt

# Set Discord bot token
export DISCORD_TOKEN='your_bot_token_here'

# Run the bot
python discord_bot.py
```

## Command-Line Tool

Enhanced CLI tool with advanced features.

### Implementation

```python
# tts_cli.py
#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import scipy.io.wavfile
from pocket_tts import TTSModel

def main():
    parser = argparse.ArgumentParser(
        description='Pocket TTS - Text-to-Speech CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --text "Hello world" --output speech.wav
  %(prog)s --text "Hi!" --voice marius --temperature 0.8
  %(prog)s --batch file.txt --output-dir ./audio
        """
    )

    # Input options
    parser.add_argument(
        '--text', '-t',
        type=str,
        help='Text to convert to speech'
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='File containing text to convert'
    )
    parser.add_argument(
        '--batch', '-b',
        type=str,
        help='Batch file (one text per line)'
    )

    # Voice options
    parser.add_argument(
        '--voice', '-v',
        type=str,
        default='alba',
        help='Voice to use (default: alba)'
    )
    parser.add_argument(
        '--list-voices',
        action='store_true',
        help='List available voices and exit'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output.wav',
        help='Output audio file (default: output.wav)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for batch processing'
    )

    # Generation parameters
    parser.add_argument(
        '--temperature',
        type=float,
        help='Generation temperature (0.0-2.0)'
    )
    parser.add_argument(
        '--decode-steps',
        type=int,
        help='Number of decode steps (1-16)'
    )

    # Performance options
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Enable torch.compile for faster inference'
    )
    parser.add_argument(
        '--threads',
        type=int,
        help='Number of CPU threads to use'
    )

    # Other options
    parser.add_argument(
        '--verbose', '-V',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    args = parser.parse_args()

    # Handle --list-voices
    if args.list_voices:
        voices = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
        print("Available voices:")
        for voice in voices:
            print(f"  - {voice}")
        return 0

    # Validate input
    if not any([args.text, args.file, args.batch]):
        parser.error("One of --text, --file, or --batch is required")

    # Initialize model
    if args.verbose:
        print("Loading TTS model...")

    model_kwargs = {}
    if args.temperature is not None:
        model_kwargs['temp'] = args.temperature
    if args.decode_steps is not None:
        model_kwargs['lsd_decode_steps'] = args.decode_steps

    model = TTSModel.load_model(**model_kwargs)

    if args.compile:
        if args.verbose:
            print("Compiling model...")
        model.compile_for_inference(mode="reduce-overhead")

    # Load voice
    if args.verbose:
        print(f"Loading voice: {args.voice}")

    voice_state = model.get_state_for_audio_prompt(args.voice)

    # Process input
    if args.text:
        # Single text input
        return generate_single(model, voice_state, args.text, args.output, args.verbose)
    elif args.file:
        # File input
        text = Path(args.file).read_text()
        return generate_single(model, voice_state, text, args.output, args.verbose)
    elif args.batch:
        # Batch processing
        return generate_batch(model, voice_state, args.batch, args.output_dir, args.verbose)

    return 0

def generate_single(model, voice_state, text, output_path, verbose):
    """Generate single audio file."""
    if verbose:
        print(f"Generating audio for: {text[:50]}...")

    audio = model.generate_audio(voice_state, text)

    if verbose:
        duration = audio.shape[0] / model.sample_rate
        print(f"Generated {duration:.2f}s of audio")

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(str(output_path), model.sample_rate, audio.numpy())

    if verbose:
        print(f"Saved to: {output_path}")

    return 0

def generate_batch(model, voice_state, batch_file, output_dir, verbose):
    """Generate multiple audio files from batch file."""
    batch_file = Path(batch_file)
    output_dir = Path(output_dir) if output_dir else batch_file.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = batch_file.read_text().strip().split('\n')

    if verbose:
        print(f"Processing {len(texts)} texts...")

    for i, text in enumerate(texts, 1):
        if not text.strip():
            continue

        if verbose:
            print(f"[{i}/{len(texts)}] {text[:50]}...")

        try:
            audio = model.generate_audio(voice_state, text)
            output_path = output_dir / f"output_{i:03d}.wav"
            scipy.io.wavfile.write(str(output_path), model.sample_rate, audio.numpy())

            if verbose:
                print(f"  -> {output_path}")

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)

    if verbose:
        print(f"Batch processing complete: {output_dir}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### Usage Examples

```bash
# Make executable
chmod +x tts_cli.py

# Single text
./tts_cli.py --text "Hello world" --output speech.wav

# From file
./tts_cli.py --file input.txt --output speech.wav

# Batch processing
./tts_cli.py --batch texts.txt --output-dir ./audio

# Custom parameters
./tts_cli.py --text "Hi!" --voice marius --temperature 0.8 --decode-steps 8

# With compilation
./tts_cli.py --text "Hello" --compile

# List voices
./tts_cli.py --list-voices
```

For more information, see:
- [Python API Documentation](python-api.md)
- [Configuration Guide](configuration-guide.md)
- [Error Handling Guide](error-handling-guide.md)
