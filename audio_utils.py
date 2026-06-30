"""
audio_utils.py — Audio loading, feature extraction, filler word detection,
and waveform generation using Librosa and SoundFile.
"""

import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use


FILLER_WORDS = {"um", "uh", "like", "you know", "basically", "literally",
                "actually", "so", "right", "okay", "well", "i mean"}

# silence threshold — frames below this RMS are considered pauses
SILENCE_THRESHOLD = 0.01


def load_audio(file_obj):
    """
    Load audio from a file path or Streamlit UploadedFile.
    Returns (audio_array, sample_rate).
    """
    if hasattr(file_obj, "read"):
        suffix = os.path.splitext(file_obj.name)[-1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_obj.read())
            tmp_path = tmp.name
        file_obj.seek(0)
    else:
        tmp_path = file_obj

    audio, sr = librosa.load(tmp_path, sr=None, mono=True)

    if hasattr(file_obj, "read"):
        os.remove(tmp_path)

    return audio, sr


def extract_audio_features(audio_path):
    """
    Extract pause ratio and RMS energy from audio.
    Returns a dict with pause_ratio and rms_energy.
    """
    audio, sr = load_audio(audio_path)

    # RMS energy per frame
    rms_frames = librosa.feature.rms(y=audio)[0]
    rms_energy = float(np.mean(rms_frames))

    # frames below threshold counted as pauses
    silent_frames = np.sum(rms_frames < SILENCE_THRESHOLD)
    pause_ratio = float(silent_frames / len(rms_frames)) if len(rms_frames) > 0 else 0.0

    return {
        "pause_ratio": round(pause_ratio, 4),
        "rms_energy": round(rms_energy, 6),
    }


def filler_word_ratio(transcript):
    """
    Compute ratio of filler words to total words in transcript.
    """
    if not transcript:
        return 0.0

    words = transcript.lower().split()
    total = len(words)
    if total == 0:
        return 0.0

    filler_count = sum(1 for w in words if w.strip(",.?!") in FILLER_WORDS)
    return round(filler_count / total, 4)


def save_waveform(audio_path):
    """
    Plot and save the audio waveform as a PNG.
    Returns the temp file path to the saved image.
    """
    audio, sr = load_audio(audio_path)

    fig, ax = plt.subplots(figsize=(10, 2))
    time_axis = np.linspace(0, len(audio) / sr, num=len(audio))
    ax.plot(time_axis, audio, color="#1f77b4", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Audio Waveform")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=100)
    plt.close(fig)

    return tmp.name
