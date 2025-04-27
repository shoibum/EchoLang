# EchoLang – GPU‑Accelerated Multilingual Speech ↔ Text ↔ Speech

**English · हिंदी · ಕನ್ನಡ**

EchoLang is a final‑year CSE project that runs *entirely on the Apple‑Silicon GPU* (Metal/MPS) and lets you:

* transcribe speech,  
* translate it between English, Hindi, and Kannada,  
* speak the result back — all in one click via a Gradio web UI.

Unlike the earlier Whisper build, this edition uses **Meta Seamless M4T‑Large (v2)** — a **dense** model that avoids the sparse‑tensor crash on M‑series Macs and matches Whisper‑large accuracy.

---

## 🚀 What’s under the hood?

| Stage | Model | Size | Runs on |
|-------|-------|------|---------|
| **Speech → Text** | `facebook/seamless-m4t-v2-large` | ≈ 1.5 GB | **MPS** (GPU) |
| **Text → Text** | IndicTrans‑2 1 B (EN↔HI / EN↔KN) | ≈ 2 GB | CPU / MPS |
| **Text → Speech** | `coqui/XTTS‑v2` | ≈ 2.4 GB | MPS |
| **UI** | Gradio 4.44 | — | Safari / Chrome |

_Total first‑run download ≈ 6 GB; cached in `./models/` + `~/.cache/hf`._

---

## ✨ Features

* **Speech‑to‑Text (STT)** — Seamless M4T recognises 100 + languages; we expose EN · HI · KN out‑of‑the‑box.
* **Machine Translation** — IndicTrans‑2 yields SOTA Hindi ↔ Kannada quality via English fallback.
* **Text‑to‑Speech (TTS)** — XTTS‑v2 produces natural voices in all three languages.
* **Speech Translation** — One click: speech → translated text → synthesised speech.
* **Gradio Web UI** — five tabs:
  1. Speech → Text  
  2. Text → Speech  
  3. Text Translation  
  4. Speech → Translated Text  
  5. Speech → Translated Speech

---

## 🖥  Requirements

* **macOS 13+** on Apple Silicon (M‑series) or any Linux/Windows PC with ≥ 8 GB RAM.  
* **Python 3.11** recommended.  
* **Disk**: ≈ 10 GB free for models & cache.  
* **Internet** only on first launch (models download once).

---

## 🔧 Installation & Run

```bash
# 1  Clone
$ git clone https://github.com/shoibum/EchoLang.git
$ cd EchoLang

# 2  Run the one‑shot installer (creates .venv if missing)
$ chmod +x setup.sh
$ ./setup.sh           # downloads Seamless M4T, IndicTrans‑2, XTTS‑v2

# 3  Launch the web app
$ source .venv/bin/activate
$ python -m src.web.app --share   # omit --share for local‑only
```
The browser opens at **http://localhost:7860** (plus a public URL if you used `--share`).

---

## 📂 Project layout (v2)

```
src/
 ├─ stt/
 │    ├─ m4t_asr.py      # Seamless M4T wrapper (GPU)
 │    └─ stt.py          # generic STT adapter
 ├─ translation/         # IndicTrans‑2 logic
 ├─ tts/                 # XTTS‑v2 wrapper
 ├─ pipeline.py          # connects STT → MT → TTS
 └─ web/app.py           # Gradio interface
models/                  # cached checkpoints  (git‑ignored)
setup.sh                 # installer + downloader
reset_models.sh          # wipe all caches
```

---

## 🛠  Behind the scenes

1. **STT** — audio → Seamless M4T encoder/decoder → text  (GPU).
2. **MT**  — text → IndicTrans‑2 translation  (CPU/GPU).
3. **TTS** — translated text → XTTS‑v2 → WAV  (GPU).
4. **UI**  — Gradio routes files/bytes between the three stages.

```
Speech  ─┐
         ├─► 1. STT ─► 2. (optional) MT ─► 3. (optional) TTS ─► Audio/Text out
Text    ─┘
```

---

## 🧹  House‑keeping

```bash
# remove every cached model (~6 GB)
$ ./reset_models.sh
```

Need a smaller footprint? Edit `MODEL_ID` in `src/stt/m4t_asr.py` to `"facebook/seamless-m4t-medium"` (≈ 800 MB) and re‑run `setup.sh`.

---

## 📄 Licence

* **Code**: MIT © 2025 shoibum
* **Seamless M4T** weights: Meta research licence  
* **IndicTrans‑2 & XTTS‑v2**: CC‑BY 4.0