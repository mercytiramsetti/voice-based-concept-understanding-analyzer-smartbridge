"""
VBCUA — Voice-Based Concept Understanding Analyser
Main Streamlit application — UI, input handling, session state, output rendering.
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # suppress torchvision-related warnings

import streamlit as st

from audio_utils import extract_audio_features, filler_word_ratio, save_waveform
from speech_to_text import speech_to_text
from semantic_eval import semantic_similarity
from scoring_engine import evaluate_understanding
from report_generator import generate_report
from reference_concepts import get_concept_names, get_reference_text


# --- page config ---
st.set_page_config(
    page_title="VBCUA",
    page_icon="🎙️",
    layout="wide",
)


# --- custom CSS ---
st.markdown("""
    <style>
        body, .stApp { background-color: #0e1117; color: #ffffff; }
        .main-title {
            font-size: 2.2rem; font-weight: 700;
            color: #ffffff; text-align: center; margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1rem; color: #a0aec0;
            text-align: center; margin-bottom: 2rem;
        }
        .concept-box {
            background-color: #1a202c; border-radius: 10px;
            padding: 1.2rem; border: 1px solid #2d3748;
        }
        .concept-title { font-size: 1.2rem; font-weight: 600; color: #ffffff; margin-bottom: 0.6rem; }
        .concept-text { font-size: 0.95rem; color: #cbd5e0; line-height: 1.6; }
        .info-bar {
            background-color: #1a365d; border-radius: 8px;
            padding: 0.8rem 1.2rem; color: #90cdf4;
            font-size: 0.95rem; margin-top: 1rem;
        }
        .section-header {
            font-size: 1.1rem; font-weight: 600;
            color: #e2e8f0; margin-bottom: 0.5rem;
        }
        .score-label { font-size: 0.85rem; color: #a0aec0; margin-bottom: 0.2rem; }
        .score-value { font-size: 2.4rem; font-weight: 700; color: #ffffff; }
        .metric-value { font-size: 1.6rem; font-weight: 600; color: #ffffff; }
        .metric-label { font-size: 0.8rem; color: #a0aec0; }
    </style>
""", unsafe_allow_html=True)


# --- initialise session state ---
for key in ("transcript", "audio_features", "filler", "similarity",
            "score", "level", "color", "waveform_img", "concept_name",
            "reference_text", "analysis_done"):
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state["analysis_done"] is None:
    st.session_state["analysis_done"] = False


# --- header ---
st.markdown('<div class="main-title">Voice-Based Concept Understanding Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated evaluation of spoken conceptual explanations using AI.</div>', unsafe_allow_html=True)
st.markdown("---")


# --- two-column input layout ---
col_upload, col_reference = st.columns([1.5, 1])

with col_upload:
    st.markdown('<div class="section-header">Upload Student Audio (WAV)</div>', unsafe_allow_html=True)
    audio_file = st.file_uploader(
        label="Audio file",
        type=["wav", "mp3"],
        help="Limit 200MB • WAV, MP3",
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-header" style="margin-top:1rem;">Select Concept</div>', unsafe_allow_html=True)
    concept_name = st.selectbox(
        label="Concept",
        options=get_concept_names(),
        label_visibility="collapsed",
    )

with col_reference:
    reference_text = get_reference_text(concept_name)
    st.markdown(f"""
        <div class="concept-box">
            <div class="concept-title">Concept Reference</div>
            <div class="concept-text">{reference_text}</div>
        </div>
    """, unsafe_allow_html=True)


# --- audio player + waveform on upload ---
if audio_file:
    st.markdown("---")
    st.audio(audio_file, format=f"audio/{audio_file.name.split('.')[-1]}")
    try:
        waveform_path = save_waveform(audio_file)
        st.image(waveform_path, use_container_width=True)
        st.session_state["waveform_img"] = waveform_path
    except Exception as e:
        st.warning(f"Could not render waveform: {e}")


# --- analyse button ---
st.markdown("")
analyse_clicked = st.button("Analyze Concept Understanding", use_container_width=True)

if not audio_file:
    st.markdown('<div class="info-bar">Upload an audio file to begin analysis.</div>', unsafe_allow_html=True)

elif analyse_clicked:
    # validate file before pipeline
    try:
        import soundfile as sf
        import io as _io
        audio_file.seek(0)
        sf.read(_io.BytesIO(audio_file.read()))
        audio_file.seek(0)
    except Exception:
        st.error("The uploaded file appears corrupted or unsupported. Please upload a valid WAV or MP3 file.")
        st.stop()

    with st.spinner("Processing and evaluating..."):
        try:
            transcript = speech_to_text(audio_file)
            audio_features = extract_audio_features(audio_file)
            filler = filler_word_ratio(transcript)
            similarity = semantic_similarity(transcript, reference_text)
            score, level, color = evaluate_understanding(similarity, filler, audio_features)

            st.session_state.update({
                "transcript": transcript,
                "audio_features": audio_features,
                "filler": filler,
                "similarity": similarity,
                "score": score,
                "level": level,
                "color": color,
                "concept_name": concept_name,
                "reference_text": reference_text,
                "analysis_done": True,
            })
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()


# --- output rendering ---
if st.session_state.get("analysis_done"):
    s = st.session_state

    st.markdown("---")
    st.success("Analysis Completed")

    # row 1: transcript | final evaluation
    col_transcript, col_eval = st.columns([1.5, 1])

    with col_transcript:
        st.markdown("### Transcribed Explanation")
        st.write(s["transcript"] or "—")

    with col_eval:
        st.markdown("### Final Evaluation")
        st.markdown('<div class="score-label">Understanding Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="score-value">{s["score"]}/100</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:1.6rem; font-weight:700; color:{s["color"]}; margin-top:0.5rem;">'
            f'{s["level"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # row 2: three metrics
    col_sim, col_filler, col_energy = st.columns(3)
    with col_sim:
        st.markdown('<div class="metric-label">Semantic Similarity</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{round(s["similarity"], 4)}</div>', unsafe_allow_html=True)
    with col_filler:
        st.markdown('<div class="metric-label">Filler Word Ratio</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{round(s["filler"], 4)}</div>', unsafe_allow_html=True)
    with col_energy:
        st.markdown('<div class="metric-label">Confidence (Energy)</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{round(s["audio_features"]["rms_energy"], 4)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # PDF download
    pdf_bytes = generate_report(
        concept_name=s["concept_name"],
        reference_text=s["reference_text"],
        transcript=s["transcript"],
        similarity=s["similarity"],
        filler=s["filler"],
        audio_features=s["audio_features"],
        score=s["score"],
        level=s["level"],
        waveform_img_path=s["waveform_img"],
    )
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="vbcua_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
