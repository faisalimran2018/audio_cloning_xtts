# XTTS Chunked Dubbing Pipeline

This repo contains a simple pipeline to:
1) split a source audio + speaker-labeled text into **chunks** and a **manifest**  
2) run **XTTS** (or other TTS) per chunk from the manifest  
3) optionally generate **English TTS** from a specific manifest text key  
4) **assemble** generated chunk audio back into one final file

---

## Quick Start (Windows)

### 1) Create + activate virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies
If you have a `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Inputs

You need:
- **Audio file** (example): `ramadan.wav`
- **Speaker/translation text file** (example): `ramadan_ur_en_ar_speakers.txt`

> Keep large audio files out of Git. See **Git / Large Files** below.

---

## Pipeline

### Step A — Make chunks + manifest
Creates chunk files and a JSON manifest used by the TTS scripts.

```bash
python make_chunks_and_manifest.py ^
  --audio .\ramadan.wav ^
  --txt .\ramadan_ur_en_ar_speakers.txt ^
  --chunks_dir ramadan_chunks ^
  --manifest ramadan_manifest.json
```

**Outputs**
- `ramadan_chunks/` (chunked assets)
- `ramadan_manifest.json` (metadata for TTS)

---

### Step B — Run XTTS from manifest (Arabic example)
Generates TTS audio for each chunk described in the manifest.

```bash
python run_xtts_from_manifest.py ^
  --manifest .\ramadan_manifest.json ^
  --device cuda ^
  --language ar ^
  --max_chars 160 ^
  --skip_existing
```

**Flags**
- `--device cuda` uses GPU (recommended). Use `--device cpu` if needed.
- `--skip_existing` avoids regenerating chunks that already exist.
- `--max_chars 160` limits per-chunk text length (tune for quality/speed).

---

### Step C — Generate English TTS (optional)
If your manifest contains English text under a key such as `en_text`, run:

```bash
python generate_english_tts.py ^
  --manifest ramadan_manifest.json ^
  --language en ^
  --text_key en_text ^
  --device cuda
```

> Make sure `--manifest` matches the manifest you created (e.g., `ramadan_manifest.json`).

---

### Step D — Assemble final audio from generated chunks
After TTS produces per-chunk audio (commonly under a `tts/` folder), assemble them:

```bash
python assemble_from_tts_folder.py --manifest ramadan_manifest.json
```

**Output**
- A combined/final audio file (exact filename/location depends on `assemble_from_tts_folder.py`).

---

## Example (end-to-end)

```bash
.venv\Scripts\activate

python make_chunks_and_manifest.py --audio .\ramadan.wav --txt .\ramadan_ur_en_ar_speakers.txt --chunks_dir ramadan_chunks --manifest ramadan_manifest.json

python run_xtts_from_manifest.py --manifest .\ramadan_manifest.json --device cuda --language ar --max_chars 160 --skip_existing

python generate_english_tts.py --manifest ramadan_manifest.json --language en --text_key en_text --device cuda

python assemble_from_tts_folder.py --manifest ramadan_manifest.json
```

---

## Recommended Repo Layout

```
.
├── make_chunks_and_manifest.py
├── run_xtts_from_manifest.py
├── generate_english_tts.py
├── assemble_from_tts_folder.py
├── ramadan_manifest.json         # OK to commit if small
├── ramadan_chunks/               # generated (do not commit)
├── tts/                          # generated (do not commit)
├── outputs/                      # generated (do not commit)
├── data/                         # optional local inputs (do not commit)
├── README.md
└── .gitignore
```

---

## Git / Large Files

### Add a `.gitignore`
Create a `.gitignore` (or add to it):

```gitignore
# Large audio
*.wav
*.mp3
*.m4a
*.flac
*.aac
*.ogg

# Generated folders
ramadan_chunks/
tts/
outputs/
data/
inputs/
artifacts/
runs/

# Python
.venv/
__pycache__/
*.pyc
.env
.DS_Store
```

If you already added large files by mistake:
```bash
git rm --cached path/to/large_file.wav
git commit -m "Remove large audio from tracking"
```

---

## Notes / Troubleshooting

- **CUDA issues**: try CPU mode:
  ```bash
  python run_xtts_from_manifest.py --manifest ramadan_manifest.json --device cpu --language ar
  ```
- **Manifest filename mismatch**: keep it consistent across commands (e.g., always use `ramadan_manifest.json`).

---

## License
Add your preferred license (MIT/Apache-2.0/etc).
