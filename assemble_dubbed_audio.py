#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

def read_wav(path: str) -> (np.ndarray, int):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # mono mixdown
    return audio, sr

def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    # simple linear resample (good enough for alignment; if you want, we can switch to librosa)
    ratio = sr_out / sr_in
    n_out = int(round(len(x) * ratio))
    if n_out <= 1 or len(x) <= 1:
        return x
    t_in = np.linspace(0, 1, num=len(x), endpoint=False)
    t_out = np.linspace(0, 1, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest.json produced earlier")
    ap.add_argument("--out", default="dubbed.wav", help="Output dubbed wav")
    ap.add_argument("--sr", type=int, default=24000, help="Output sample rate (set to your XTTS output SR if known)")
    ap.add_argument("--gain", type=float, default=1.0, help="Gain applied to each TTS chunk")
    args = ap.parse_args()

    m: Dict[str, Any] = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    segs: List[Dict[str, Any]] = m["segments"]

    # determine total length
    end_max = 0.0
    for s in segs:
        end_max = max(end_max, float(s["end"]))
    total_samples = int(end_max * args.sr) + int(1.0 * args.sr)  # +1s tail

    out = np.zeros(total_samples, dtype=np.float32)

    missing = 0
    for s in segs:
        wav_path = Path(s["tts_out_wav"])
        if not wav_path.exists():
            missing += 1
            continue

        x, sr_in = read_wav(str(wav_path))
        x = resample_linear(x, sr_in, args.sr)

        start = float(s["start"])
        start_i = int(round(start * args.sr))
        end_i = min(len(out), start_i + len(x))
        if start_i >= len(out):
            continue

        chunk = x[: max(0, end_i - start_i)] * float(args.gain)
        out[start_i:end_i] += chunk

    # avoid clipping
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak

    sf.write(args.out, out, args.sr, subtype="FLOAT")
    print(f"✅ Exported: {args.out}")
    if missing:
        print(f"⚠️ Missing {missing} generated TTS wav files (skipped).")

if __name__ == "__main__":
    main()
