import os

from report_generator import generate_report
from audio_utils import save_waveform


def test_generate_report_returns_valid_pdf_bytes(sine_wav_path):
    waveform_path = save_waveform(sine_wav_path)
    try:
        pdf_bytes = generate_report(
            concept_name="Machine Learning",
            reference_text="Machine Learning is a subset of artificial intelligence.",
            transcript="Machine learning lets systems learn patterns from data.",
            similarity=0.87,
            filler=0.05,
            audio_features={"pause_ratio": 0.12, "rms_energy": 0.045},
            score=90,
            level="Strong Understanding",
            waveform_img_path=waveform_path,
        )
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF")
        assert len(pdf_bytes) > 1000
    finally:
        os.remove(waveform_path)


def test_generate_report_handles_missing_transcript_and_waveform():
    pdf_bytes = generate_report(
        concept_name="Blockchain",
        reference_text="Blockchain is a distributed ledger technology.",
        transcript="",
        similarity=0.0,
        filler=0.0,
        audio_features={},
        score=10,
        level="Poor Understanding",
        waveform_img_path=None,
    )
    assert pdf_bytes.startswith(b"%PDF")
