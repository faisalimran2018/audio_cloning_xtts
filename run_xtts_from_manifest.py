#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import List

import torch
import soundfile as sf
import numpy as np
from TTS.api import TTS


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def split_text_max_chars(text: str, max_chars: int) -> List[str]:
    """
    Split text into chunks <= max_chars, preferring sentence/phrase boundaries.
    Works for Arabic too (؟ ، . ! etc.)
    """
    text = normalize_spaces(text)
    if len(text) <= max_chars:
        return [text]

    # First split by strong punctuation boundaries
    parts = re.split(r"(?<=[۔?!؟.!])\s+|(?<=[،,;:])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]

    chunks: List[str] = []
    cur = ""
    for p in parts:
        if not cur:
            cur = p
            continue
        if len(cur) + 1 + len(p) <= max_chars:
            cur = f"{cur} {p}"
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)

    # If any chunk is still too long (no punctuation), hard-split by words
    final: List[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
            continue
        words = c.split(" ")
        buf = ""
        for w in words:
            if not buf:
                buf = w
            elif len(buf) + 1 + len(w) <= max_chars:
                buf = f"{buf} {w}"
            else:
                final.append(buf)
                buf = w
        if buf:
            final.append(buf)

    return [x for x in final if x.strip()]


def read_wav(path: Path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr


def write_wav(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sr, subtype="FLOAT")


def concat_wavs(wavs: List[Path], out_path: Path):
    if not wavs:
        raise ValueError("No wavs to concat")
    audios = []
    sr0 = None
    for w in wavs:
        a, sr = read_wav(w)
        if sr0 is None:
            sr0 = sr
        elif sr != sr0:
            raise ValueError(f"Sample-rate mismatch while concatenating: {w} is {sr} but expected {sr0}")
        audios.append(a)
    merged = np.concatenate(audios) if audios else np.zeros(0, dtype=np.float32)
    write_wav(out_path, merged, sr0 or 24000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest.json produced earlier")
    ap.add_argument("--model", default="tts_models/multilingual/multi-dataset/xtts_v2")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--language", default="ar", help="Arabic is 'ar'")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--min_chars", type=int, default=2)
    ap.add_argument("--max_chars", type=int, default=160, help="Max chars per TTS call (set slightly below 166)")
    ap.add_argument("--pause_ms", type=int, default=120, help="Silence between parts when stitching (ms)")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    segments = manifest["segments"]

    # Load XTTS once
    tts = TTS(args.model)
    if args.device == "cuda" and torch.cuda.is_available():
        tts.to("cuda")
    else:
        tts.to("cpu")

    ok, skipped, failed = 0, 0, 0

    for s in segments:
        text_ar = normalize_spaces(s.get("text_ar") or "")
        speaker_wav = Path(s["chunk_wav"])
        out_wav = Path(s["tts_out_wav"])

        if len(text_ar) < args.min_chars:
            skipped += 1
            continue
        if not speaker_wav.exists():
            print(f"Missing speaker chunk: {speaker_wav}")
            failed += 1
            continue
        if args.skip_existing and out_wav.exists():
            skipped += 1
            continue

        out_wav.parent.mkdir(parents=True, exist_ok=True)

        # Split long Arabic text into parts A/B/C...
        parts = split_text_max_chars(text_ar, args.max_chars)
        part_wavs: List[Path] = []

        try:
            if len(parts) == 1:
                # normal
                tts.tts_to_file(
                    text=parts[0],
                    file_path=str(out_wav),
                    speaker_wav=str(speaker_wav),
                    language=args.language,
                )
                ok += 1
                print(f"[OK] {s['id']} -> {out_wav.name}")
                continue

            # multi-part: generate each part then concat
            for idx, p in enumerate(parts, start=1):
                pw = out_wav.with_name(f"{out_wav.stem}__part{idx:02d}.wav")
                tts.tts_to_file(
                    text=p,
                    file_path=str(pw),
                    speaker_wav=str(speaker_wav),
                    language=args.language,
                )
                part_wavs.append(pw)

            # Add small pauses between parts (optional but helps naturalness)
            audios = []
            sr0 = None
            pause = None

            for j, pw in enumerate(part_wavs):
                a, sr = read_wav(pw)
                if sr0 is None:
                    sr0 = sr
                    pause = np.zeros(int(sr0 * (args.pause_ms / 1000.0)), dtype=np.float32)
                audios.append(a)
                if j != len(part_wavs) - 1 and pause is not None:
                    audios.append(pause)

            merged = np.concatenate(audios) if audios else np.zeros(0, dtype=np.float32)
            write_wav(out_wav, merged, sr0 or 24000)

            ok += 1
            print(f"[OK] {s['id']} -> {out_wav.name} (parts={len(parts)})")

        except Exception as e:
            failed += 1
            print(f"[FAIL] {s.get('id','?')}: {e}")

    print(f"\nDone. OK={ok}, Skipped={skipped}, Failed={failed}")
    print("Next: assemble timeline audio using assemble_dubbed_audio.py -> final_audio.wav")


if __name__ == "__main__":
    main()
