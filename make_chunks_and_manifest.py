#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

SAMPLE_RATE = 16000  # whisperx.load_audio returns 16k mono float32


TS_RE = re.compile(r"\[(?P<spk>[^\]]+)\]\s+(?P<start>\d\d:\d\d:\d\d\.\d{3})\s+-\s+(?P<end>\d\d:\d\d:\d\d\.\d{3})")
UR_RE = re.compile(r"^UR:\s*(.*)$")
EN_RE = re.compile(r"^EN:\s*(.*)$")
ARUR_RE = re.compile(r"^AR\(UR\):\s*(.*)$")
AREN_RE = re.compile(r"^AR\(EN\):\s*(.*)$")


def ts_to_seconds(ts: str) -> float:
    # HH:MM:SS.mmm
    h, m, rest = ts.split(":")
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def load_audio_16k_mono(audio_path: str) -> np.ndarray:
    # Prefer whisperx loader (handles many formats via ffmpeg)
    import whisperx
    return whisperx.load_audio(audio_path)


def slice_audio(audio: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    s = int(max(0.0, start_s) * SAMPLE_RATE)
    e = int(max(0.0, end_s) * SAMPLE_RATE)
    return audio[s:e]


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write float32 WAV
    import soundfile as sf
    sf.write(str(path), audio.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")


def parse_transcript_txt(txt_path: Path) -> List[Dict[str, Any]]:
    """
    Parses blocks like:
      [SPEAKER_00] 00:00:05.347 - 00:00:33.848
      UR: ...
      EN: ...
      AR(UR): ...
      AR(EN): ...
    """
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    segments: List[Dict[str, Any]] = []

    cur: Optional[Dict[str, Any]] = None
    for line in lines:
        line = line.strip()
        if not line:
            # end of block
            if cur:
                segments.append(cur)
                cur = None
            continue

        m = TS_RE.match(line)
        if m:
            if cur:
                segments.append(cur)
            cur = {
                "speaker": m.group("spk").strip(),
                "start_ts": m.group("start"),
                "end_ts": m.group("end"),
                "start": ts_to_seconds(m.group("start")),
                "end": ts_to_seconds(m.group("end")),
                "ur_text": "",
                "en_text": "",
                "ar_from_ur": "",
                "ar_from_en": "",
            }
            continue

        if not cur:
            continue

        mu = UR_RE.match(line)
        if mu:
            cur["ur_text"] = mu.group(1).strip()
            continue
        me = EN_RE.match(line)
        if me:
            cur["en_text"] = me.group(1).strip()
            continue
        maru = ARUR_RE.match(line)
        if maru:
            cur["ar_from_ur"] = maru.group(1).strip()
            continue
        maren = AREN_RE.match(line)
        if maren:
            cur["ar_from_en"] = maren.group(1).strip()
            continue

    if cur:
        segments.append(cur)

    # keep only segments with Arabic-from-English (your XTTS input)
    out = []
    for s in segments:
        if s["ar_from_en"].strip():
            out.append(s)
    return out


def safe_filename(s: str) -> str:
    s = s.replace(":", "-")
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Original audio (same file used in WhisperX)")
    ap.add_argument("--txt", required=True, help="Transcript file (UR/EN/AR(UR)/AR(EN))")
    ap.add_argument("--chunks_dir", default="chunks", help="Where to save timestamped chunks")
    ap.add_argument("--tts_dir", default="tts", help="Where XTTS will write generated wavs")
    ap.add_argument("--manifest", default="manifest.json", help="Output manifest JSON")
    ap.add_argument("--pad", type=float, default=0.05, help="Pad seconds added before/after chunk (helps XTTS)")
    ap.add_argument("--min_dur", type=float, default=0.40, help="Skip segments shorter than this duration")
    args = ap.parse_args()

    txt_path = Path(args.txt)
    audio_path = Path(args.audio)
    chunks_dir = Path(args.chunks_dir)
    tts_dir = Path(args.tts_dir)

    segments = parse_transcript_txt(txt_path)
    audio = load_audio_16k_mono(str(audio_path))

    manifest: Dict[str, Any] = {
        "audio": str(audio_path.resolve()),
        "sample_rate": SAMPLE_RATE,
        "chunks_dir": str(chunks_dir.resolve()),
        "tts_dir": str(tts_dir.resolve()),
        "segments": []
    }

    for i, seg in enumerate(segments):
        start = float(seg["start"])
        end = float(seg["end"])
        dur = max(0.0, end - start)
        if dur < args.min_dur:
            continue

        # pad (clamped)
        ps = max(0.0, start - args.pad)
        pe = min(len(audio) / SAMPLE_RATE, end + args.pad)

        chunk = slice_audio(audio, ps, pe)

        spk = seg["speaker"]
        start_ts = seg["start_ts"]
        end_ts = seg["end_ts"]

        seg_id = f"seg_{i:06d}"
        chunk_name = f"{seg_id}__{spk}__{safe_filename(start_ts)}__{safe_filename(end_ts)}.wav"
        chunk_path = chunks_dir / spk / chunk_name

        write_wav(chunk_path, chunk)

        tts_out = tts_dir / f"{seg_id}.wav"

        manifest["segments"].append({
            "id": seg_id,
            "speaker": spk,
            "start": start,
            "end": end,
            "duration": dur,
            "chunk_wav": str(chunk_path),
            "tts_out_wav": str(tts_out),
            "text_ar": seg["ar_from_en"],  # << use AR(EN) for XTTS input
            # optional debug fields:
            "ur_text": seg["ur_text"],
            "en_text": seg["en_text"],
            "ar_from_ur": seg["ar_from_ur"],
        })

    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote chunks to: {chunks_dir}/")
    print(f"✅ Wrote manifest: {args.manifest}")
    print("Next: run XTTS for each manifest segment -> write to tts_out_wav paths.")


if __name__ == "__main__":
    main()
