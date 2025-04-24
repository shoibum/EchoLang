# Multilingual Speech-to-Text and Text-to-Speech System

A comprehensive multilingual speech-to-text and text-to-speech system supporting English, Hindi, and Kannada with translation capabilities. Designed as a final‑year Computer Science Engineering project, this system:

- **Automatically falls back** from Apple MPS (Metal) to CPU when sparse‑tensor ops are unsupported.  
- Uses **Whisper** for general ASR, and a **Kannada‑fine‑tuned Wav2Vec2** model to produce native Devanagari output.  
- Provides **text translation** via Hugging Face Transformers: English ↔ Hindi (Helsinki‑NLP) and English ↔ Kannada (AI4Bharat IndicTrans2).  
- Offers **speech translation** (speech → translated text → synthesized speech).  
- Ships with a **Gradio** web interface for effortless demos.

---

## 🚀 Features

- **Speech‑to‑Text (STT)**  
  - Whisper (`tiny`, `base`, `small`, etc.) with automatic MPS→CPU fallback on Apple Silicon  
  - Native Kannada ASR via Wav2Vec2 fine‑tuned for Devanagari output  
- **Text‑to‑Speech (TTS)**  
  - Coqui TTS in English, Hindi, and Kannada (handles both separate‑vocoder and bundled‑vocoder models)  
- **Machine Translation**  
  - English ↔ Hindi: `Helsinki‑NLP/opus‑mt‑en‑hi`  
  - English ↔ Kannada: `ai4bharat/indictrans2‑en‑indic‑1B` & vice versa  
  - Automatic fallback for Hindi ↔ Kannada via English  
- **Speech Translation**  
  - Speech in one language → translated text → synthesized speech in target language  
- **Web UI**  
  - Gradio app with tabs:  
    - Speech → Text  
    - Text → Speech  
    - Text Translation  
    - Speech → Translated Text  
    - Speech → Translated Speech  

---

## 🛠 Technologies Used

- **Python 3.8+**  
- **OpenAI Whisper** (ASR)  
- **Facebook Wav2Vec2** fine‑tuned for Kannada (Devanagari output)  
- **Hugging Face Transformers** (translation pipelines & ASR models)  
- **Coqui TTS** (text‑to‑speech)  
- **Gradio** (web interface)  
- **PyTorch** (deep learning backend)  
- **soundfile**, **librosa** (audio I/O & resampling)  
- **indic-transliteration** (optional, for post‑processing romanized Hindi/Kannada)  
- **Git** (version control)  

---

## 💻 System Requirements

- **OS**: macOS (Apple Silicon or Intel), Linux, or Windows  
- **RAM**: ≥ 8 GB (16 GB+ recommended)  
- **Disk**: ≥ 10 GB free  
- **Python**: 3.8+  
- **Internet**: required on first run to download pre‑trained models  

---

## 🔧 Installation

```bash
# 1. Clone
git clone https://github.com/shoibum/EchoLang.git
cd EchoLang

# 2. Create & activate venv
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
