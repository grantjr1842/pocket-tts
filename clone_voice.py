import os

import scipy.io.wavfile

from pocket_tts import TTSModel

# Define paths
VOICE_PATH = "assets/voice-references/tars/tars-voice-sample-01.wav"
OUTPUT_PATH = "tars_clone.wav"


def main():
    print("Loading TTS Model...")
    # Load the model
    tts_model = TTSModel.load_model()

    print(f"Loading voice state from: {VOICE_PATH}")
    if not os.path.exists(VOICE_PATH):
        print(f"Error: Voice file not found at {VOICE_PATH}")
        return

    # Get voice state from an audio file
    voice_state = tts_model.get_state_for_audio_prompt(VOICE_PATH)

    text_to_generate = "Hello, this is Tars. I am ready to assist you."
    print(f"Generating audio for text: '{text_to_generate}'")

    # Generate audio
    audio = tts_model.generate_audio(voice_state, text_to_generate)

    print(f"Saving output to: {OUTPUT_PATH}")
    # Save to file
    scipy.io.wavfile.write(OUTPUT_PATH, tts_model.sample_rate, audio.numpy())
    print("Done!")


if __name__ == "__main__":
    main()
