import os
import json
import uuid
import tempfile
import subprocess

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse

from pydub import AudioSegment
from TTS.api import TTS

# ============================================
# CONFIG
# ============================================

OUTPUT_DIR = "outputs"
TEMP_DIR = "temp"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Load XTTS once
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

app = FastAPI(title="Local Dubbing API")

# ============================================
# HELPERS
# ============================================

def stretch_audio(input_path, output_path, target_duration):
    """
    Stretch/compress audio to target duration using ffmpeg atempo.
    """

    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]

    current_duration = float(
        subprocess.check_output(probe_cmd).decode().strip()
    )

    speed = current_duration / target_duration

    # ffmpeg atempo only supports 0.5-2.0
    filters = []

    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0

    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5

    filters.append(f"atempo={speed}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-filter:a", ",".join(filters),
        output_path
    ]

    subprocess.run(cmd, check=True)


def overlay_segment(base_audio, segment_audio, start_ms):
    return base_audio.overlay(segment_audio, position=start_ms)


# ============================================
# MAIN API
# ============================================

@app.post("/dub")
async def dub_audio(
    audio: UploadFile = File(...),
    diarization: UploadFile = File(...)
):

    job_id = str(uuid.uuid4())

    # Save uploaded audio
    original_audio_path = os.path.join(
        TEMP_DIR,
        f"{job_id}_original.wav"
    )

    with open(original_audio_path, "wb") as f:
        f.write(await audio.read())

    # Load diarization JSON
    diarization_data = json.loads(
        await diarization.read()
    )

    # Load original audio
    original_audio = AudioSegment.from_file(original_audio_path)

    total_duration = len(original_audio)

    # Silent base
    final_audio = AudioSegment.silent(duration=total_duration)

    # Speaker map
    speaker_map = diarization_data["speakers"]

    # ============================================
    # PROCESS EACH SEGMENT
    # ============================================

    for idx, segment in enumerate(diarization_data["segments"]):

        speaker = segment["speaker"]
        text = segment["text"]

        start = float(segment["start"])
        end = float(segment["end"])

        duration = end - start

        speaker_wav = speaker_map[speaker]

        raw_tts_path = os.path.join(
            TEMP_DIR,
            f"{job_id}_{idx}_raw.wav"
        )

        aligned_tts_path = os.path.join(
            TEMP_DIR,
            f"{job_id}_{idx}_aligned.wav"
        )

        # Generate TTS
        tts.tts_to_file(
            text=text,
            file_path=raw_tts_path,
            speaker_wav=speaker_wav,
            language="ar"
        )

        # Match duration
        stretch_audio(
            raw_tts_path,
            aligned_tts_path,
            duration
        )

        # Load generated chunk
        generated_segment = AudioSegment.from_file(
            aligned_tts_path
        )

        # Overlay at exact timestamp
        final_audio = overlay_segment(
            final_audio,
            generated_segment,
            int(start * 1000)
        )

    # ============================================
    # EXPORT
    # ============================================

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{job_id}_dubbed.wav"
    )

    final_audio.export(output_path, format="wav")

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="dubbed.wav"
    )