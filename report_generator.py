"""
report_generator.py — PDF report generation using ReportLab.
Layout: Reference Concept → Student Transcription → Audio Visualization → Evaluation Summary table.
"""

import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable
)


PAGE_W, PAGE_H = A4
MARGIN = 20 * mm


def _styles():
    base = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=base["Title"],
        fontSize=18,
        textColor=colors.HexColor("#1a202c"),
        spaceAfter=4,
    )
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=base["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#2d3748"),
        spaceBefore=12,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#4a5568"),
        leading=15,
        spaceAfter=6,
    )
    return title_style, section_style, body_style


def generate_report(concept_name, reference_text, transcript,
                    similarity, filler, audio_features, score, level, waveform_img_path):
    """
    Build a PDF report and return it as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )

    title_style, section_style, body_style = _styles()
    story = []

    # report title
    story.append(Paragraph("Voice-Based Concept Understanding Analyser", title_style))
    story.append(Paragraph("Evaluation Report", body_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 6))

    # reference concept
    story.append(Paragraph("Reference Concept", section_style))
    story.append(Paragraph(f"<b>{concept_name}</b>", body_style))
    story.append(Paragraph(reference_text, body_style))
    story.append(Spacer(1, 4))

    # student transcription
    story.append(Paragraph("Student Transcription", section_style))
    story.append(Paragraph(transcript or "No transcription available.", body_style))
    story.append(Spacer(1, 4))

    # audio visualization
    if waveform_img_path:
        story.append(Paragraph("Audio Visualization", section_style))
        img = Image(waveform_img_path, width=160 * mm, height=40 * mm)
        story.append(img)
        story.append(Spacer(1, 6))

    # evaluation summary table
    story.append(Paragraph("Evaluation Summary", section_style))
    table_data = [
        ["Metric", "Value"],
        ["Semantic Similarity", str(round(similarity, 4))],
        ["Filler Word Ratio", str(round(filler, 4))],
        ["Pause Ratio", str(audio_features.get("pause_ratio", "—"))],
        ["Confidence (Energy)", str(audio_features.get("rms_energy", "—"))],
        ["Final Score", f"{score}/100"],
        ["Understanding Level", level],
    ]
    table = Table(table_data, colWidths=[90 * mm, 80 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#2d3748")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 11),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7fafc"), colors.white]),
        ("FONTSIZE",    (0, 1), (-1, -1), 10),
        ("TEXTCOLOR",   (0, 1), (-1, -1), colors.HexColor("#2d3748")),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)

    doc.build(story)
    return buffer.getvalue()
