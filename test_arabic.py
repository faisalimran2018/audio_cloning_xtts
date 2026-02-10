from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

tts.tts_to_file(
    text="مرحبا! هذا نموذج دبلجة يعمل محلياً على ويندوز بصوت شهباز.",
    file_path="out_ar.wav",
    speaker_wav="Shah.wav",
    language="ar"  # <-- Arabic
)

print("Wrote out_ar.wav")
