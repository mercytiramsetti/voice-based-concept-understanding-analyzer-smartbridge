import io
import os

from audio_utils import extract_audio_features, filler_word_ratio, save_waveform, load_audio


# ---- filler_word_ratio -----------------------------------------------

def test_filler_word_ratio_empty_transcript_is_zero():
    assert filler_word_ratio("") == 0.0
    assert filler_word_ratio(None) == 0.0


def test_filler_word_ratio_no_fillers_is_zero():
    assert filler_word_ratio("Machine learning improves with more data") == 0.0


def test_filler_word_ratio_counts_known_filler_words():
    # "so", "um", "like", "basically" are fillers -> 4 of 9 words
    transcript = "so um machine learning is like basically pattern recognition"
    ratio = filler_word_ratio(transcript)
    assert ratio == round(4 / 9, 4)


def test_filler_word_ratio_strips_punctuation_before_matching():
    transcript = "So, machine learning is basically great."
    ratio = filler_word_ratio(transcript)
    # "So," and "basically." should still match after stripping punctuation
    assert ratio == round(2 / 6, 4)


# ---- extract_audio_features -------------------------------------------

def test_extract_audio_features_returns_expected_keys(sine_wav_path):
    features = extract_audio_features(sine_wav_path)
    assert set(features.keys()) == {"pause_ratio", "rms_energy"}
    assert 0.0 <= features["pause_ratio"] <= 1.0
    assert features["rms_energy"] >= 0.0


def test_extract_audio_features_tone_has_higher_energy_than_silence(sine_wav_path, silence_wav_path):
    tone_features = extract_audio_features(sine_wav_path)
    silence_features = extract_audio_features(silence_wav_path)
    assert tone_features["rms_energy"] > silence_features["rms_energy"]
    assert silence_features["pause_ratio"] > tone_features["pause_ratio"]


def test_extract_audio_features_accepts_streamlit_like_uploaded_file(sine_wav_path):
    with open(sine_wav_path, "rb") as f:
        raw_bytes = f.read()

    class FakeUploadedFile(io.BytesIO):
        name = "recording.wav"

    uploaded = FakeUploadedFile(raw_bytes)
    features = extract_audio_features(uploaded)
    assert features["rms_energy"] >= 0.0
    # the fake uploaded-file object must be left readable for later use
    uploaded.seek(0)
    assert uploaded.read() == raw_bytes


# ---- save_waveform -------------------------------------------------------

def test_save_waveform_produces_a_png_file(sine_wav_path):
    image_path = save_waveform(sine_wav_path)
    try:
        assert os.path.exists(image_path)
        assert image_path.endswith(".png")
        assert os.path.getsize(image_path) > 0
    finally:
        os.remove(image_path)


# ---- load_audio -----------------------------------------------------------

def test_load_audio_from_path_returns_array_and_sample_rate(sine_wav_path):
    audio, sr = load_audio(sine_wav_path)
    assert sr == 16000
    assert len(audio) > 0
