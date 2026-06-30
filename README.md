# Voice-Based Concept Understanding Analyser (VBCUA)

An AI-powered web application that evaluates spoken conceptual explanations using speech-to-text transcription, semantic similarity analysis, and audio feature extraction.

---

## Overview

VBCUA helps students, educators, and trainers assess how well a concept is understood through spoken explanation. Upload an audio file, select a concept, and the system automatically transcribes, analyses, and scores the explanation — then generates a downloadable PDF report.

---

## Features

- **Speech Transcription** — Converts uploaded audio to text using OpenAI Whisper (runs locally, no API key needed)
- **Semantic Similarity** — Compares explanation against a reference concept using Sentence-BERT
- **Audio Analysis** — Extracts pause ratio, RMS energy, and filler word ratio using Librosa
- **Comprehension Scoring** — Combines all metrics into a score (0–100) with classification: Strong / Moderate / Poor Understanding
- **Waveform Visualization** — Displays the audio waveform after upload
- **PDF Report** — Downloadable report with transcript, waveform image, metrics table, and evaluation summary

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Speech-to-Text | OpenAI Whisper (`base` model) |
| Semantic Similarity | Sentence-BERT (`all-MiniLM-L6-v2`) |
| Audio Processing | Librosa, SoundFile |
| Visualization | Matplotlib |
| PDF Generation | ReportLab |
| Language | Python 3.10+ |

---

## Project Structure

```
vbcua-smartbridge/
├── app.py                  # Streamlit entry point
├── speech_to_text.py       # Whisper transcription
├── semantic_eval.py        # Sentence-BERT similarity
├── audio_utils.py          # Audio features, filler words, waveform
├── scoring_engine.py       # Score calculation and classification
├── report_generator.py     # PDF report generation
├── reference_concepts.py   # Predefined concept reference texts
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd vbcua-smartbridge
```

### 2. Create and activate virtual environment

```bash
python3 -m venv vbcu_env
source vbcu_env/bin/activate        # Mac/Linux
vbcu_env\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How It Works

```
Audio Upload → Whisper Transcription → Sentence-BERT Similarity
                                              ↓
                           Audio Features (Librosa) + Filler Word Detection
                                              ↓
                                     Scoring Engine
                                              ↓
                              Results Display + PDF Report
```

### Scoring Breakdown

| Metric | Condition | Points |
|--------|-----------|--------|
| Semantic Similarity | > 0.7 | 50 |
| Semantic Similarity | 0.4 – 0.7 | 30 |
| Semantic Similarity | < 0.4 | 10 |
| Filler Word Ratio | < 0.05 | 20 |
| Filler Word Ratio | ≥ 0.05 | 10 |
| Pause Ratio | < 0.25 | 15 |
| Pause Ratio | ≥ 0.25 | 5 |
| RMS Energy | > 0.01 | 15 |
| RMS Energy | ≤ 0.01 | 5 |

| Score | Classification |
|-------|---------------|
| ≥ 80 | Strong Understanding |
| 50 – 79 | Moderate Understanding |
| < 50 | Poor Understanding |

---

## Supported Concepts

- Machine Learning
- Cloud Computing
- Deep Learning
- Natural Language Processing
- Computer Vision
- Blockchain
- Internet of Things
- Cybersecurity

---

## Generating a Test Audio (Mac)

```bash
# Good explanation (targets ~100 score)
say -o ~/Downloads/ml_good.wav --data-format=LEF32@22050 \
  "Machine learning is a subset of artificial intelligence that allows systems \
  to learn patterns from data and improve performance without being explicitly programmed."

# Average explanation (targets ~60-70 score)
say -o ~/Downloads/ml_average.wav --data-format=LEF32@22050 \
  "So um, machine learning is basically like when computers learn stuff from data. \
  Like its a part of AI and um it helps systems get better at things without someone \
  actually programming every single step you know what I mean."
```

---

## Requirements

```
openai-whisper
sentence-transformers
torch
librosa
soundfile
audioread
nltk
numpy
matplotlib
streamlit
reportlab
pytest
```

---

## Notes

- Whisper `base` model is downloaded automatically on first run (~150MB)
- Sentence-BERT model is downloaded automatically on first run (~90MB)
- Both models run entirely offline after the initial download
- Supported audio formats: WAV, MP3
