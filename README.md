# EchoLang – Multilingual Speech Pipeline

**English · हिंदी · ಕನ್ನಡ**

EchoLang is a versatile project that translates speech and text between English, Hindi, and Kannada. It runs locally on your CPU.

## ✨ Features

* **Speech-to-Text (STT)** — Uses FasterWhisper models (fine-tuned for Hindi & Kannada, base for English) via CTranslate2 for efficient transcription (CPU).
* **Machine Translation** — Uses local AI4Bharat IndicTrans2 distilled models for high-quality translation between English, Hindi, and Kannada (CPU).
* **Text-to-Speech (TTS)** — Uses Coqui XTTS-v2 (for English/Hindi) and Facebook MMS-TTS (for Kannada) to produce natural voices (CPU).
* **Speech Translation** — Provides end-to-end flows: speech → translated text → synthesized speech.
* **Gradio Web UI** — Simple interface with tabs for different functions:
    1.  Speech → Text
    2.  Text → Speech
    3.  Text Translation
    4.  Speech → Translated Text
    5.  Speech → Translated Speech

## 🖥 Requirements

* **OS:** macOS, Linux, or Windows (setup script requires bash, e.g., Git Bash/WSL on Windows).
* **Python:** 3.11+ recommended.
* **Disk:** ≈ 5-10 GB free space (for downloaded/converted models and cache).
* **RAM:** ≥ 8 GB recommended (more may be needed depending on models loaded).
* **CPU:** All models are configured to run on the CPU.
* **Internet:** Required only for the first run to download models and dependencies.

## 📂 Project Layout (Current)

```
models/
├─ base-small-ct2/     # Converted FasterWhisper Base model
├─ hindi-small-ct2/    # Converted FasterWhisper Hindi model
├─ kannada-small-ct2/  # Converted FasterWhisper Kannada model
└─ xtts_v2/            # Downloaded XTTSv2 model files
src/
├─ stt/
│    ├─ faster_whisper_asr.py # FasterWhisper model wrapper
│    └─ stt.py                # STT interface (loads multiple models)
├─ translation/
│    └─ translator.py         # Translation interface (uses IndicTrans2)
├─ tts/
│    ├─ mms_tts.py            # MMS-TTS model wrapper (for Kannada)
│    ├─ synthesizer.py        # TTS interface (routes to XTTS/MMS)
│    └─ xtts.py               # XTTS-v2 model wrapper (for Eng/Hin)
├─ utils/
│    ├─ audio.py
│    ├─ language.py
│    └─ model_utils.py
├─ web/
│    ├─ app.py                # Gradio UI definition
│    └─ components.py         # Gradio tab definitions
├─ __init__.py
├─ config.py               # Central configuration
└─ pipeline.py             # End-to-end pipeline logic
main.py                    # Main application entry point
requirements.txt           # Python dependencies
setup.sh                   # Setup script
reset_models.sh            # Script to reset some local models
```

## 🚀 Quick Start

1.  Clone the repository:
    ```bash
    # git clone <your-repo-url>
    # cd <your-repo-directory>
    ```

2.  Run the setup script (make it executable first):
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

3.  Activate the virtual environment:
    * macOS/Linux: `source venv/bin/activate`
    * Windows (Git Bash/WSL): `source venv/Scripts/activate`
    * Windows (CMD): `venv\Scripts\activate.bat`
    * Windows (PowerShell): `.\venv\Scripts\Activate.ps1`

4.  **(Included in setup.sh)** Install the IndicTransToolkit separately (if setup.sh didn't):
    ```bash
    pip install git+[https://github.com/VarunGumma/IndicTransToolkit.git](https://github.com/VarunGumma/IndicTransToolkit.git)
    ```

5.  Convert the required FasterWhisper STT models into the `./models/` directory:
    ```bash
    # Ensure ctranslate2 is installed: pip install "ctranslate2>=3.0.0"
    ct2-transformers-converter --model vasista22/whisper-kannada-small --output_dir models/kannada-small-ct2 --quantization int8 --copy_files preprocessor_config.json tokenizer_config.json
    ct2-transformers-converter --model vasista22/whisper-hindi-small --output_dir models/hindi-small-ct2 --quantization int8 --copy_files preprocessor_config.json tokenizer_config.json
    ct2-transformers-converter --model openai/whisper-small --output_dir models/base-small-ct2 --quantization int8 --copy_files preprocessor_config.json tokenizer_config.json
    ```

6.  Launch the application:
    ```bash
    python main.py
    ```

7.  Open your browser to the local URL provided (usually http://127.0.0.1:7860).

## 🛠 Behind the Scenes

1.  **STT:** Uploaded audio → Correct FasterWhisper model (kn/hi/en) selected → Transcription (CPU).
2.  **MT:** Text → Correct IndicTrans2 model selected (En->Indic, Indic->En, Indic->Indic) → Translation (CPU).
3.  **TTS:** Text → Correct TTS model selected (MMS for kn, XTTS for en/hi) → Synthesized WAV audio (CPU).
4.  **UI:** Gradio handles file uploads/inputs and displays outputs.

```
Speech ──► 1. STT (FasterWhisper)
       ──► 2. (Optional) MT (IndicTrans2)
       ──► 3. (Optional) TTS (XTTS/MMS)
       ──► Audio Output

Text ──► 2. (Optional) MT ──► 3. (Optional) TTS ──► Audio Output

```

## 📚 Model References

* FasterWhisper: [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) (using fine-tuned Whisper models by [vasista22](https://huggingface.co/vasista22) and base models by [openai](https://huggingface.co/openai))
* IndicTrans2: [AI4Bharat/IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) (using distilled models)
* IndicTransToolkit: [VarunGumma/IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit)
* XTTS-v2: [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) (via [coqui-ai/TTS](https://github.com/coqui-ai/TTS))
* MMS-TTS: [facebook/mms-tts-kan](https://huggingface.co/facebook/mms-tts-kan) (via `transformers`)

```
