# EchoLang – GPU‑Accelerated Multilingual Speech ↔ Text ↔ Speech

**English · हिंदी · ಕನ್ನಡ**

EchoLang is a final‑year CSE project that runs *entirely on the Apple‑Silicon GPU* (Metal/MPS) and lets you:
* Transcribe speech,
* Translate it between English, Hindi, and Kannada,
* Speak the result back — all in one click via a Gradio web UI.

## ✨ Features

* **Speech‑to‑Text (STT)** — Seamless M4T recognizes 100+ languages; we expose EN · HI · KN out‑of‑the‑box.
* **Machine Translation** — IndicTrans‑2 yields SOTA Hindi ↔ Kannada quality via English fallback.
* **Text‑to‑Speech (TTS)** — XTTS‑v2 produces natural voices in all three languages.
* **Speech Translation** — One click: speech → translated text → synthesized speech.
* **Gradio Web UI** — Five tabs:
   1. Speech → Text
   2. Text → Speech
   3. Text Translation
   4. Speech → Translated Text
   5. Speech → Translated Speech

## 🖥 Requirements

* **macOS 13+** on Apple Silicon (M‑series) or any Linux/Windows PC with ≥ 8 GB RAM.
* **Python 3.11** recommended.
* **Disk**: ≈ 10 GB free for models & cache.
* **Internet** only on first launch (models download once).

## 📂 Project Layout

```
src/
 ├─ stt/
 │    ├─ m4t_asr.py        # Seamless M4T ASR implementation
 │    └─ stt.py            # Speech-to-text interface
 ├─ translation/
 │    ├─ indictrans.py     # IndicTrans-2 implementation
 │    └─ translator.py     # Translation interface
 ├─ tts/
 │    ├─ xtts.py           # XTTS-v2 implementation
 │    └─ synthesizer.py    # Text-to-speech interface
 ├─ utils/
 │    ├─ audio.py          # Audio processing utilities
 │    ├─ model_utils.py    # Model loading/caching utilities
 │    └─ language.py       # Language detection and code mapping
 ├─ pipeline.py            # End-to-end pipeline implementation
 └─ web/
      ├─ app.py            # Gradio web UI
      └─ components.py     # UI components and layouts
models/                    # Model storage directory
requirements.txt           # Python dependencies
setup.sh                   # Setup script
reset_models.sh            # Script to reset/redownload models
```

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/username/echolang.git
   cd echolang
   ```

2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Launch the application:
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   python -m src.web.app
   ```

4. Open your browser at http://localhost:7860

## 🛠 Behind the Scenes

1. **STT** — audio → Seamless M4T encoder/decoder → text (GPU).
2. **MT** — text → IndicTrans‑2 translation (CPU/GPU).
3. **TTS** — translated text → XTTS‑v2 → WAV (GPU).
4. **UI** — Gradio routes files/bytes between the three stages.

```
Speech  ─┐
         ├─► 1. STT ─► 2. (optional) MT ─► 3. (optional) TTS ─► Audio/Text out
Text    ─┘
```

## 📚 Model References

- Seamless M4T: [FAIR Meta AI](https://github.com/facebookresearch/fairseq/tree/main/examples/seamless_communication)
- IndicTrans2: [AI4Bharat](https://github.com/AI4Bharat/IndicTrans2)
- XTTS-v2: [Coqui TTS](https://github.com/coqui-ai/TTS)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.