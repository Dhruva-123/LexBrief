from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_analysis_pdf(records, document_id):
    """
    Generates a PDF bytes representation of the rhetorical roles analysis report.
    """
    buffer = BytesIO()
    
    # Setup document: letter size, 40pt margins
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles matching application aesthetics
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=8
    )
    
    meta_style = ParagraphStyle(
        name='MetaStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=15
    )
    
    body_style = ParagraphStyle(
        name='BodyStyle',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )
    
    role_style = ParagraphStyle(
        name='RoleStyle',
        parent=styles['Normal'],
        fontSize=9,
        fontWeight='Bold',
        textColor=colors.HexColor('#2563eb')
    )
    
    story = []
    
    # Header Section
    story.append(Paragraph("Legal Judgment Rhetorical Role Analysis", title_style))
    story.append(Paragraph(f"Document ID: {document_id} | Total Sentences: {len(records)}", meta_style))
    story.append(Spacer(1, 10))
    
    # Table headers
    table_data = [
        [
            Paragraph("<b>#</b>", body_style),
            Paragraph("<b>Rhetorical Role</b>", body_style),
            Paragraph("<b>Sentence Text</b>", body_style)
        ]
    ]
    
    # Populate sentences and roles
    for idx, rec in enumerate(records):
        role_str = rec.get("role") or "None"
        num_p = Paragraph(str(idx + 1), body_style)
        role_p = Paragraph(f"<b>{role_str}</b>", role_style)
        text_p = Paragraph(rec.get("sentence", ""), body_style)
        table_data.append([num_p, role_p, text_p])
        
    # Table layout: col widths sum up to 532 (available width is 612 - 80 margins)
    t = Table(table_data, colWidths=[32, 120, 380])
    
    t_style = TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#334155')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8fafc')),
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor('#cbd5e1')),
        ('LINEBELOW', (0, 1), (-1, -1), 0.5, colors.HexColor('#f1f5f9')),
    ])
    
    # Alternating row background shading
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            t_style.add('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f8fafc'))
            
    t.setStyle(t_style)
    story.append(t)
    
    # Build document PDF
    doc.build(story)
    
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data


def generate_summary_pdf(document_id, filename, model_name, summary_text):
    """
    Generates a PDF bytes representation of the legal judgment summarization report.
    """
    if isinstance(summary_text, dict):
        text_parts = []
        if "final_summary" in summary_text:
            text_parts.append(f"COMBINED SUMMARY:\n{summary_text['final_summary']}")
        if "section_summaries" in summary_text:
            sec_sums = summary_text["section_summaries"]
            text_parts.append(
                f"SECTION SUMMARIES:\n"
                f"- Facts: {sec_sums.get('facts', '')}\n"
                f"- Analysis & Arguments: {sec_sums.get('analysis', '')}\n"
                f"- Conclusion: {sec_sums.get('conclusion', '')}"
            )
        if "sentence_counts" in summary_text and "chunk_counts" in summary_text:
            text_parts.append(
                f"STATISTICS:\n"
                f"- Facts: {summary_text['sentence_counts'].get('facts', 0)} sentences, {summary_text['chunk_counts'].get('facts', 0)} chunks\n"
                f"- Analysis & Arguments: {summary_text['sentence_counts'].get('analysis', 0)} sentences, {summary_text['chunk_counts'].get('analysis', 0)} chunks\n"
                f"- Conclusion: {summary_text['sentence_counts'].get('conclusion', 0)} sentences, {summary_text['chunk_counts'].get('conclusion', 0)} chunks"
            )
        summary_text = "\n\n".join(text_parts)

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=8
    )
    
    meta_style = ParagraphStyle(
        name='MetaStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=15
    )
    
    body_style = ParagraphStyle(
        name='BodyStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=10
    )
    
    story = []
    story.append(Paragraph("Legal Judgment Summarization Report", title_style))
    story.append(Paragraph(f"Document: {filename}<br/>Model: {model_name}<br/>Document ID: {document_id}", meta_style))
    story.append(Spacer(1, 10))
    
    # Process text by paragraphs
    for paragraph in summary_text.split("\n\n"):
        if paragraph.strip():
            text_escaped = paragraph.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
            story.append(Paragraph(text_escaped, body_style))
            
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data


def generate_explanation_pdf(document_id, filename, predicted_judgment, explanation_passages):
    """
    Generates a PDF bytes representation of the CJPE judgment prediction and explanation report.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=8
    )
    
    meta_style = ParagraphStyle(
        name='MetaStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=15
    )
    
    section_title_style = ParagraphStyle(
        name='SectionTitleStyle',
        parent=styles['Heading2'],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=6,
        spaceBefore=12
    )

    body_style = ParagraphStyle(
        name='BodyStyle',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )
    
    pred_style = ParagraphStyle(
        name='PredStyle',
        parent=styles['Normal'],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#1e3a8a'),
        spaceAfter=10
    )
    
    story = []
    story.append(Paragraph("Legal Judgment Prediction &amp; Explanation Report", title_style))
    story.append(Paragraph(f"Document: {filename}<br/>Document ID: {document_id}", meta_style))
    story.append(Spacer(1, 10))
    
    # Prediction result
    story.append(Paragraph(f"<b>Predicted Judgment Outcome:</b> {predicted_judgment}", pred_style))
    story.append(Spacer(1, 5))
    
    # Table headers
    story.append(Paragraph("<b>Top Explanation Sentences (Reasoning)</b>", section_title_style))
    story.append(Spacer(1, 4))
    
    table_data = [
        [
            Paragraph("<b>#</b>", body_style),
            Paragraph("<b>Page</b>", body_style),
            Paragraph("<b>Rhetorical Role</b>", body_style),
            Paragraph("<b>Score</b>", body_style),
            Paragraph("<b>Sentence Text</b>", body_style)
        ]
    ]
    
    for idx, passage in enumerate(explanation_passages):
        num_p = Paragraph(str(idx + 1), body_style)
        page_val = passage.get("page_number")
        page_p = Paragraph(str(page_val) if page_val is not None else "-", body_style)
        role_p = Paragraph(passage.get("role") or "None", body_style)
        score_val = passage.get("importance_score", 0.0)
        score_p = Paragraph(f"{score_val:.4f}", body_style)
        sentence_val = passage.get("sentence") or ""
        text_p = Paragraph(sentence_val.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), body_style)
        table_data.append([num_p, page_p, role_p, score_p, text_p])
        
    t = Table(table_data, colWidths=[25, 35, 92, 50, 330])
    t_style = TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#334155')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8fafc')),
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor('#cbd5e1')),
        ('LINEBELOW', (0, 1), (-1, -1), 0.5, colors.HexColor('#f1f5f9')),
    ])
    
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            t_style.add('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f8fafc'))
            
    t.setStyle(t_style)
    story.append(t)
    
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data
