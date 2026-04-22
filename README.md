# EchoLang — Multilingual Speech Pipeline

**English · हिंदी · ಕನ್ನಡ**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-CPU%20Only-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

EchoLang is a locally deployable multilingual speech pipeline that composes speech recognition, machine translation, and speech synthesis into a single CPU-only application supporting English, Hindi, and Kannada. No GPU, no cloud, no API keys required after initial model setup.

---

## Why EchoLang

Most state-of-the-art speech models require GPU infrastructure — typically 8–24 GB VRAM — that is economically inaccessible to individual users and small organizations in low-resource Indian language communities. Existing tools for Indian languages are also fragmented across different research groups, model families, and APIs, with no unified system offering end-to-end speech translation for languages like Kannada.

EchoLang addresses this by demonstrating that careful architectural decomposition — model quantization, lazy loading, and strategy-pattern routing — enables a complete three-stage speech pipeline to operate within 8 GB RAM on a consumer CPU, with no cloud dependency and full data privacy.

---

## Features

- **Speech-to-Text (STT)** — FasterWhisper (CTranslate2) with language-specific fine-tuned models for Hindi and Kannada, and base Whisper Small for English. int8 quantized for CPU efficiency.
- **Machine Translation (MT)** — AI4Bharat IndicTrans2 distilled models (200M parameters) handling six directional language pairs (en↔hi, en↔kn, hi↔kn) through a unified API.
- **Text-to-Speech (TTS)** — Strategy-pattern routing between Coqui XTTS-v2 (English/Hindi) and Facebook MMS-TTS (Kannada), selected per language based on output quality and runtime compatibility.
- **Five Workflows** — Composable pipeline supporting increasing depths of processing:
  1. Speech → Text
  2. Text → Speech
  3. Text → Translated Text
  4. Speech → Translated Text
  5. Speech → Translated Speech
- **Gradio Web UI** — Tabbed interface exposing all five workflows with audio upload, text input, and language selectors.
- **Fully Local** — No internet connection required after initial model downloads. All inference runs on-device.

---

## Architecture

EchoLang follows a three-tier layered architecture:

```
┌─────────────────────────────────────────────────────┐
│               Presentation Layer                    │
│         Gradio Web UI — 5 workflow tabs             │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│            Pipeline Orchestration Layer             │
│   pipeline.py — routing, composition, error handling│
└──────┬──────────────┬──────────────────┬────────────┘
       │              │                  │
┌──────▼──────┐ ┌─────▼──────┐ ┌────────▼───────────┐
│ STT Layer   │ │  MT Layer  │ │     TTS Layer       │
│FasterWhisper│ │IndicTrans2 │ │ XTTS-v2 / MMS-TTS  │
│ (en/hi/kn)  │ │(6 lang prs)│ │  (en,hi) / (kn)    │
└─────────────┘ └────────────┘ └────────────────────┘
```

| Subsystem | Backend | Languages | Format |
|-----------|---------|-----------|--------|
| STT | FasterWhisper (CTranslate2) | English, Hindi, Kannada | int8 quantized |
| MT | IndicTrans2 distilled 200M | en↔hi, en↔kn, hi↔kn | PyTorch |
| TTS | XTTS-v2 / MMS-TTS | en, hi / kn | PyTorch |

---

## Performance

All measurements on consumer hardware (8 GB RAM, CPU only, no GPU).

| Workflow | First Run (incl. model load) | Subsequent Runs |
|----------|------------------------------|-----------------|
| Speech → Text (10s audio) | 15–25s | 3–8s |
| Text → Speech (1 sentence) | 20–40s | 5–15s |
| Text Translation | 10–20s | 2–5s |
| Speech → Translated Text | 25–45s | 5–12s |
| Speech → Translated Speech | 40–75s | 10–25s |

| Component | Peak RAM | Notes |
|-----------|----------|-------|
| Baseline (app only) | ~200 MB | Python + Gradio + dependencies |
| + STT (1 Whisper model) | ~500 MB | int8 quantized CTranslate2 |
| + MT (1 IndicTrans2 model) | ~1.2 GB | Distilled 200M + tokenizer |
| + TTS (XTTS-v2) | ~2.8 GB | Largest single component |
| Full pipeline peak | ~4.5–6 GB | All models loaded simultaneously |

First-run latency is dominated by model loading. Subsequent runs show 3–5x improvement once models are cached in memory. The full pipeline fits within 8 GB with headroom for the operating system.

---

## Requirements

- **OS:** macOS, Linux, or Windows (setup script requires bash; use Git Bash or WSL on Windows)
- **Python:** 3.11+ recommended
- **RAM:** 8 GB minimum (all models loaded simultaneously peak at ~4.5–6 GB)
- **Disk:** ~5–10 GB free space for downloaded and converted models
- **CPU:** All inference runs on CPU — no GPU required
- **Internet:** Required only for the first run to download models and dependencies

---

## Project Structure

```
EchoLang/
├── src/
│   ├── stt/
│   │   ├── faster_whisper_asr.py   # FasterWhisper model wrapper
│   │   └── stt.py                  # STT interface (language routing)
│   ├── translation/
│   │   └── translator.py           # IndicTrans2 wrapper (6 lang pairs)
│   ├── tts/
│   │   ├── synthesizer.py          # TTS interface (strategy routing)
│   │   ├── xtts.py                 # XTTS-v2 wrapper (English/Hindi)
│   │   └── mms_tts.py              # MMS-TTS wrapper (Kannada)
│   ├── web/
│   │   ├── app.py                  # Gradio UI definition
│   │   └── components.py           # Gradio tab definitions
│   ├── utils/
│   │   ├── audio.py
│   │   ├── language.py
│   │   └── model_utils.py
│   ├── config.py                   # Centralised model paths and settings
│   ├── pipeline.py                 # End-to-end pipeline orchestration
│   └── __init__.py
├── models/
│   ├── base-small-ct2/             # Converted FasterWhisper English model
│   ├── hindi-small-ct2/            # Converted FasterWhisper Hindi model
│   ├── kannada-small-ct2/          # Converted FasterWhisper Kannada model
│   └── xtts_v2/                    # Downloaded XTTS-v2 model files
├── recordings/                     # Sample audio files for testing
├── main.py                         # Application entry point
├── requirements.txt
├── setup.sh                        # Environment setup script
├── reset_models.sh                 # Script to clear cached models
└── README.md
```

---

## Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/shoibum/EchoLang
cd EchoLang
```

**2. Run the setup script**
```bash
chmod +x setup.sh
./setup.sh
```

This will create a virtual environment and install all dependencies including IndicTransToolkit.

**3. Activate the virtual environment**
```bash
# macOS / Linux
source venv/bin/activate

# Windows (Git Bash / WSL)
source venv/Scripts/activate

# Windows (CMD)
venv\Scripts\activate.bat

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

**4. Convert Whisper models to CTranslate2 format**

Run these once to download and convert the STT models into the `./models/` directory:
```bash
ct2-transformers-converter --model vasista22/whisper-kannada-small \
  --output_dir models/kannada-small-ct2 --quantization int8 \
  --copy_files preprocessor_config.json tokenizer_config.json

ct2-transformers-converter --model vasista22/whisper-hindi-small \
  --output_dir models/hindi-small-ct2 --quantization int8 \
  --copy_files preprocessor_config.json tokenizer_config.json

ct2-transformers-converter --model openai/whisper-small \
  --output_dir models/base-small-ct2 --quantization int8 \
  --copy_files preprocessor_config.json tokenizer_config.json
```

**5. Launch the application**
```bash
python main.py
```

Open your browser at **http://127.0.0.1:7860**

---

## How It Works

```
Audio Input
    │
    ▼
[STT] FasterWhisper selects model by language (en/hi/kn)
    │  → int8 quantized CTranslate2 inference on CPU
    │  → transcribed text
    │
    ▼ (optional)
[MT] IndicTrans2 selects directional model
    │  → En→Indic / Indic→En / Indic→Indic
    │  → translated text
    │
    ▼ (optional)
[TTS] Strategy routing by target language
    │  → XTTS-v2 for English / Hindi
    │  → MMS-TTS for Kannada
    │  → WAV output (22050 Hz, 16-bit PCM)
    │
    ▼
Audio / Text Output (Gradio UI)
```

---

## Limitations

- **No real-time streaming** — all processing is batch-mode; the complete audio input must be provided before transcription begins.
- **No automatic language detection** — the user must specify the input language. Code-mixed speech is not handled.
- **Kannada TTS quality** — MMS-TTS output for Kannada is intelligible but lacks the naturalness of XTTS-v2 output for English/Hindi. This reflects the current state of open-source Dravidian TTS, not an architectural limitation.
- **Cold start latency** — first use of each workflow requires model loading, which takes 15–40 seconds depending on the component.
- **No speaker diarization** — multi-speaker audio is transcribed as a single speaker.

---

## Model References

| Component | Model | Source |
|-----------|-------|--------|
| STT (Hindi/Kannada) | vasista22/whisper-hindi-small, whisper-kannada-small | [HuggingFace](https://huggingface.co/vasista22) |
| STT (English) | openai/whisper-small | [HuggingFace](https://huggingface.co/openai/whisper-small) |
| STT Runtime | faster-whisper (CTranslate2) | [GitHub](https://github.com/SYSTRAN/faster-whisper) |
| MT | AI4Bharat IndicTrans2 distilled | [GitHub](https://github.com/AI4Bharat/IndicTrans2) |
| MT Toolkit | IndicTransToolkit | [GitHub](https://github.com/VarunGumma/IndicTransToolkit) |
| TTS (en/hi) | Coqui XTTS-v2 | [HuggingFace](https://huggingface.co/coqui/XTTS-v2) |
| TTS (kn) | facebook/mms-tts-kan | [HuggingFace](https://huggingface.co/facebook/mms-tts-kan) |

---

## Academic Context

Developed as Final Year Project (21CSP76) for B.E. in Computer Science and Engineering, JSS Academy of Technical Education, Bangalore — Visvesvaraya Technological University, 2025.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
