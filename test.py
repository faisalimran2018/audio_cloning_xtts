from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

tts.tts_to_file(
    text="Hello! This is dubbing model running locally on Windows with Shahbaz voice.",
    file_path="out.wav",
    speaker_wav="speaker.wav",  # <-- put speaker.wav in same folder
    language="en"
)

print("Wrote out.wav")


# pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121