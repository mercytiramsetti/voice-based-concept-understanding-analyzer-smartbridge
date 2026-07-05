"""Shared pytest fixtures — mainly synthetic audio generation so tests
never depend on real recordings or network access."""

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def sine_wav_path(tmp_path):
    """A short, clearly-audible sine tone — used to exercise the
    RMS-energy / pause-ratio branch of audio feature extraction."""
    sr = 16000
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(path, tone, sr)
    return str(path)


@pytest.fixture
def silence_wav_path(tmp_path):
    """Near-silent audio — used to exercise the pause/low-energy branch."""
    sr = 16000
    duration = 1.0
    silence = np.zeros(int(sr * duration), dtype=np.float32)
    path = tmp_path / "silence.wav"
    sf.write(path, silence, sr)
    return str(path)
