"""
speech_to_text.py — Whisper-based audio transcription.
Handles WAV/MP3/M4A formats and normalizes audio before transcription.
"""

import os
import tempfile
import whisper
import numpy as np
import soundfile as sf


# load once at module level to avoid reloading on every call
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = whisper.load_model("base")
    return _model


def _normalize_audio(audio_array):
    # scale to [-1, 1] range for consistent whisper input
    max_val = np.max(np.abs(audio_array))
    if max_val > 0:
        return audio_array / max_val
    return audio_array


def speech_to_text(audio_path):
    """
    Transcribe audio file to text using Whisper.
    Accepts a file path string or a Streamlit UploadedFile object.
    Returns the transcribed text string.
    """
    model = _get_model()

    # if it's a streamlit uploaded file, save to a temp file first
    if hasattr(audio_path, "read"):
        suffix = os.path.splitext(audio_path.name)[-1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_path.read())
            tmp_path = tmp.name
        audio_path.seek(0)  # reset for any later reads
    else:
        tmp_path = audio_path

    try:
        # normalize before transcription
        audio_array, sample_rate = sf.read(tmp_path)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)  # stereo to mono
        audio_array = _normalize_audio(audio_array.astype(np.float32))

        # write normalized audio back to temp file for whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as norm_tmp:
            sf.write(norm_tmp.name, audio_array, sample_rate)
            norm_path = norm_tmp.name

        result = model.transcribe(norm_path)
        return result["text"].strip()

    finally:
        # clean up temp files
        if hasattr(audio_path, "read") and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if "norm_path" in locals() and os.path.exists(norm_path):
            os.remove(norm_path)
