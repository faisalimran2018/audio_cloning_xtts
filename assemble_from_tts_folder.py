#!/usr/bin/env python3
"""
Assemble dubbed segment WAVs into a single timeline WAV using manifest timestamps.

SYNC FIX:
- Places audio by manifest start time
- Fits each segment to its target window (end-start) to prevent "ends early" gaps

Fit modes:
- pad_trim    (default): trim or pad with silence (no artifacts)
- resample_fit: resample to target length (changes pitch; clamp ratios)
- none        : place as-is

Usage:
  python assemble_from_tts_folder.py --manifest manifest.json --fit_mode pad_trim --normalize
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import soundfile as sf


def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # mixdown to mono
    return audio.astype(np.float32), int(sr)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sr, subtype="FLOAT")


def resample_to_sr(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear resample to match sample rate (keeps duration correct in seconds)."""
    if sr_in == sr_out or len(x) < 2:
        return x.astype(np.float32)
    ratio = sr_out / sr_in
    n_out = int(round(len(x) * ratio))
    if n_out <= 1:
        return x.astype(np.float32)
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)


def fit_pad_trim(x: np.ndarray, target_len: int) -> np.ndarray:
    """Fit by trimming or padding zeros (no pitch/time-warp)."""
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(x) > target_len:
        return x[:target_len]
    if len(x) < target_len:
        return np.concatenate([x, np.zeros(target_len - len(x), dtype=np.float32)])
    return x


def fit_by_resample(x: np.ndarray, target_len: int) -> np.ndarray:
    """Fit by resampling to target length (changes speed AND pitch)."""
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(x) < 2 or target_len < 2:
        return fit_pad_trim(x, target_len)
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)


def resolve_existing_path(tts_dir: Path, p: str) -> Path:
    """
    Manifest paths may already include 'tts\\...' or be absolute.
    Try in this order:
      1) absolute or relative as-is (relative to current working dir)
      2) join with tts_dir
      3) if path starts with 'tts/' and joining caused 'tts/tts', strip one 'tts'
    """
    pp = Path(p)

    # 1) absolute
    if pp.is_absolute():
        return pp

    # 1b) as-is relative to CWD
    if pp.exists():
        return pp

    # 2) join with tts_dir
    cand = (tts_dir / pp)
    if cand.exists():
        return cand

    # 3) fix common double-tts: tts_dir/tts/seg.wav when tts_dir already is .../tts
    parts = pp.parts
    if len(parts) >= 1 and parts[0].lower() in ("tts", "tts_dir"):
        cand2 = tts_dir / Path(*parts[1:])
        if cand2.exists():
            return cand2

    return cand  # return best guess (used for error messages)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest.json produced earlier")
    ap.add_argument("--tts_dir", default=None, help="Override TTS folder. If omitted, uses manifest['tts_dir'] or ./tts")
    ap.add_argument("--out_name", default="final_audio.wav", help="Output filename written inside tts_dir")
    ap.add_argument("--sr", type=int, default=None, help="Output SR. If omitted, uses SR of first found wav")
    ap.add_argument("--gain", type=float, default=1.0, help="Gain multiplier applied to each segment wav")
    ap.add_argument("--normalize", action="store_true", help="Normalize final output to avoid clipping")

    ap.add_argument(
        "--fit_mode",
        default="pad_trim",
        choices=["pad_trim", "resample_fit", "none"],
        help="Fit each segment to (end-start) duration. Default pad_trim.",
    )
    ap.add_argument("--min_stretch", type=float, default=0.75, help="Min allowed target/src ratio for resample_fit")
    ap.add_argument("--max_stretch", type=float, default=1.35, help="Max allowed target/src ratio for resample_fit")
    ap.add_argument("--tail_s", type=float, default=1.0, help="Extra tail seconds added to final output")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    m: Dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    segs: List[Dict[str, Any]] = m.get("segments", [])
    if not segs:
        raise ValueError("Manifest has no segments.")

    # Determine tts_dir (where final output is written)
    if args.tts_dir:
        tts_dir = Path(args.tts_dir).expanduser()
    else:
        tts_dir = Path(m.get("tts_dir") or "tts").expanduser()
    tts_dir = tts_dir.resolve()

    if not tts_dir.exists():
        raise FileNotFoundError(f"TTS directory not found: {tts_dir}")

    # Pick output SR
    out_sr: Optional[int] = args.sr
    if out_sr is None:
        for s in segs:
            wav_rel = s.get("tts_out_wav")
            if not wav_rel:
                continue
            wp = resolve_existing_path(tts_dir, wav_rel)
            if wp.exists():
                _, sr0 = read_wav(wp)
                out_sr = sr0
                break
    if out_sr is None:
        raise FileNotFoundError(
            "Could not determine output SR (no existing segment wavs found). "
            "Provide --sr (e.g. 24000) or check tts_out_wav paths."
        )

    # Determine final length based on manifest end times
    end_max = 0.0
    for s in segs:
        if "end" in s:
            end_max = max(end_max, float(s["end"]))
    total_samples = int(round(end_max * out_sr)) + int(round(float(args.tail_s) * out_sr))
    out = np.zeros(total_samples, dtype=np.float32)

    placed = 0
    missing_files = 0
    missing_times = 0
    fit_applied = 0
    fit_fallback = 0

    for s in segs:
        if "start" not in s or "end" not in s:
            missing_times += 1
            continue

        wav_rel = s.get("tts_out_wav")
        if not wav_rel:
            missing_files += 1
            continue

        wav_path = resolve_existing_path(tts_dir, wav_rel)
        if not wav_path.exists():
            missing_files += 1
            continue

        start_s = float(s["start"])
        end_s = float(s["end"])
        if end_s <= start_s:
            continue

        start_i = int(round(start_s * out_sr))
        if start_i >= len(out):
            continue

        x, sr_in = read_wav(wav_path)
        if sr_in != out_sr:
            x = resample_to_sr(x, sr_in, out_sr)

        x = x * float(args.gain)
        target_len = int(round((end_s - start_s) * out_sr))

        if args.fit_mode != "none":
            if args.fit_mode == "pad_trim":
                x = fit_pad_trim(x, target_len)
                fit_applied += 1
            else:  # resample_fit
                ratio = target_len / max(1, len(x))
                if ratio < float(args.min_stretch) or ratio > float(args.max_stretch):
                    x = fit_pad_trim(x, target_len)
                    fit_fallback += 1
                else:
                    x = fit_by_resample(x, target_len)
                    fit_applied += 1

        end_i = min(len(out), start_i + len(x))
        if end_i > start_i:
            out[start_i:end_i] += x[: (end_i - start_i)]
            placed += 1

    if args.normalize and out.size:
        peak = float(np.max(np.abs(out)))
        if peak > 1.0:
            out = out / peak

    out_path = tts_dir / args.out_name
    write_wav(out_path, out, int(out_sr))

    print(f"✅ TTS dir: {tts_dir}")
    print(f"✅ Placed {placed} segment wav(s) into timeline")
    if missing_files:
        print(f"⚠️ Missing segment wav(s): {missing_files}")
    if missing_times:
        print(f"⚠️ Segments missing start/end: {missing_times}")
    if args.fit_mode != "none":
        print(f"✅ Fit mode: {args.fit_mode} (applied={fit_applied}, fallback_to_pad_trim={fit_fallback})")
    print(f"✅ Wrote final audio: {out_path}")


if __name__ == "__main__":
    main()
