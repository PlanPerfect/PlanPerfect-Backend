import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, TableStyle, Table, HRFlowable
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from datetime import datetime
from pathlib import Path
from Services import Logger

def safe_get(dictionary, *keys, default='Not available'):
    """Safely get nested dictionary values with a default fallback."""
    value = dictionary
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
        if value is None:
            return default
    return value if value else default

def normalize_display_list(values):
    """Flatten mixed/nested values into a clean, de-duplicated list of strings."""
    if values is None:
        return []

    if not isinstance(values, list):
        values = [values]

    cleaned = []
    seen = set()
    for value in values:
        items = value if isinstance(value, list) else [value]
        for item in items:
            if item is None:
                continue
            text = str(item).strip()
            if not text or text.lower() in {"not specified", "not available"}:
                continue
            if text not in seen:
                seen.add(text)
                cleaned.append(text)
    return cleaned

def _render_image_grid(story, image_paths, heading_style, subheading_style, body_style, caption_style,
                        section_title, intro_text, img_captions=None):
    """
    Render a list of local image paths into the PDF story as a tidy grid.
    - 1 image  → full-width centred
    - 2 images → side by side
    - 3+       → two per row
    img_captions: optional list of caption strings matching image_paths length.
    """
    valid_paths = [p for p in image_paths if p and os.path.exists(p)]
    if not valid_paths:
        return

    story.append(Paragraph(section_title, heading_style))
    if intro_text:
        story.append(Paragraph(intro_text, body_style))
    story.append(Spacer(1, 0.15 * inch))

    cell_cap_style = ParagraphStyle(
        'AgentImgCap',
        parent=body_style,
        fontSize=8,
        textColor=colors.HexColor('#666666'),
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique',
        spaceAfter=4,
    )

    def _make_img(path, width, height):
        try:
            return Image(path, width=width, height=height, kind='proportional')
        except Exception as e:
            Logger.log(f"[DESIGN DOCS] - ERROR: Could not load image {path}: {e}")
            return None

    def _caption(idx):
        if img_captions and idx < len(img_captions):
            return img_captions[idx]
        return f"Image {idx + 1}"

    if len(valid_paths) == 1:
        img = _make_img(valid_paths[0], 5.0 * inch, 3.5 * inch)
        if img:
            img_table = Table([[img], [Paragraph(_caption(0), cell_cap_style)]], colWidths=[6.5 * inch])
            img_table.setStyle(TableStyle([
                ('ALIGN',        (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
                ('BOX',          (0, 0), (0, 0), 1, colors.HexColor('#D4AF37')),
                ('TOPPADDING',   (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING',(0, 0), (-1, -1), 6),
            ]))
            story.append(img_table)

    else:
        # Two-column grid
        col_w = 3.1 * inch
        rows = []
        for i in range(0, len(valid_paths), 2):
            img_row = []
            cap_row = []
            for j in range(2):
                idx = i + j
                if idx < len(valid_paths):
                    img = _make_img(valid_paths[idx], col_w, 2.4 * inch)
                    img_row.append(img if img else Paragraph("(image unavailable)", cell_cap_style))
                    cap_row.append(Paragraph(_caption(idx), cell_cap_style))
                else:
                    img_row.append("")
                    cap_row.append("")
            rows.append(img_row)
            rows.append(cap_row)

        grid = Table(rows, colWidths=[col_w + 0.15 * inch, col_w + 0.15 * inch])
        grid.setStyle(TableStyle([
            ('ALIGN',        (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING',   (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
            ('LEFTPADDING',  (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            # Gold border on every image row (even rows)
            *[('BOX', (c, r), (c, r), 1, colors.HexColor('#D4AF37'))
              for r in range(0, len(rows), 2) for c in range(2)],
        ]))
        story.append(grid)

    story.append(Spacer(1, 0.3 * inch))


def generate_pdf(design_data, raw_floor_plan_path, segmented_floor_plan_path, preferences, budget, unit_info,
                 agent_floor_plan_paths=None, agent_generated_image_paths=None):
    """
    Generate a formatted PDF from the provided design data parameters.
    All sections are optional - if data is missing, the section will be skipped.
    This ensures the PDF can be generated even with incomplete LLM responses.

    agent_floor_plan_paths      → Agent -> Outputs -> Generated Floor Plans (list of local file paths)
    agent_generated_image_paths → Agent -> Outputs -> Generated Images (list of local file paths)
    """
    agent_floor_plan_paths      = agent_floor_plan_paths      or []
    agent_generated_image_paths = agent_generated_image_paths or []

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch
    )

    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#D4AF37'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#333333'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=13,
        textColor=colors.HexColor('#555555'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
        fontName='Helvetica'
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#666666'),
        alignment=TA_CENTER,
        spaceAfter=12,
        fontName='Helvetica-Oblique'
    )

    chatbot_cta_style = ParagraphStyle(
        'ChatbotCTA',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#D4AF37'),
        alignment=TA_CENTER,
        spaceAfter=6,
        spaceBefore=20,
        fontName='Helvetica-Bold'
    )

    highlight_box_style = ParagraphStyle(
        'HighlightBox',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#333333'),
        fontName='Helvetica'
    )

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#666666'),
        fontName='Helvetica-Oblique',
        spaceAfter=12
    )

    story = []
    logo_added = False

    # ========== LOGO AND TITLE PAGE ==========
    story.append(Spacer(1, 0.3*inch))

    BASE_DIR = Path(__file__).resolve().parents[1]
    logo_image_path = BASE_DIR / "static" / "Logo.png"
    logo_text_path = BASE_DIR / "static" / "Logo-Text.png"

    if logo_image_path.exists():
        try:
            logo_image = Image(str(logo_image_path), width=1.7 * inch, height=1.7 * inch, kind='proportional')
            page_width = letter[0] - doc.leftMargin - doc.rightMargin
            logo_table = Table([[logo_image]], colWidths=[page_width])
            logo_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(logo_table)
            story.append(Spacer(1, 0.15*inch))
            logo_added = True
        except Exception as e:
            Logger.log(f"[DESIGN DOCS] - ERROR: Could not add logo image {logo_image_path.name}: {e}")

    if logo_text_path.exists():
        try:
            logo_text_img = Image(str(logo_text_path), width=2.5*inch, height=0.6*inch, kind='proportional')
            page_width = letter[0] - doc.leftMargin - doc.rightMargin
            logo_text_table = Table([[logo_text_img]], colWidths=[page_width])
            logo_text_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(logo_text_table)
            story.append(Spacer(1, 0.3*inch))
            logo_added = True
        except Exception as e:
            Logger.log(f"[DESIGN DOCS] - ERROR: Could not add logo text PNG: {e}")

    if not logo_added:
        company_style = ParagraphStyle(
            'CompanyName',
            parent=styles['Normal'],
            fontSize=18,
            textColor=colors.HexColor('#D4AF37'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            spaceAfter=20
        )
        story.append(Paragraph("PlanPerfect", company_style))
        story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Interior Design Documentation", title_style))
    story.append(Spacer(1, 0.2*inch))

    # Property information
    if unit_info:
        unit_rooms = safe_get(unit_info, 'unit_rooms', default='Residential Unit')
        unit_sizes = normalize_display_list(safe_get(unit_info, 'unit_sizes', default=[]))
        unit_types = normalize_display_list(safe_get(unit_info, 'unit_types', default=[]))

        property_info = f"<b>Unit Type:</b> {unit_rooms}"
        story.append(Paragraph(property_info, styles['Normal']))

        if unit_types:
            story.append(Paragraph(f"<b>Unit Models:</b> {', '.join(unit_types)}", styles['Normal']))

        if unit_sizes:
            story.append(Paragraph(f"<b>Unit Sizes:</b> {', '.join(unit_sizes)}", styles['Normal']))

    if budget:
        story.append(Paragraph(f"<b>Client's Budget:</b> {budget}", styles['Normal']))

    story.append(Paragraph(f"<b>Generated On:</b> {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))

    # ========== QUOTATION RANGE SECTION ==========
    if design_data and design_data.get('quotation_range'):
        quotation_range = design_data['quotation_range']
        story.append(Paragraph("Quotation ", subheading_style))

        min_quote = safe_get(quotation_range, 'minimum_quote', default='TBD')
        max_quote = safe_get(quotation_range, 'maximum_quote', default='TBD')
        recommended_quote = safe_get(quotation_range, 'recommended_quote', default='TBD')
        quote_basis = safe_get(quotation_range, 'quote_basis', default='')
        scope_level = safe_get(quotation_range, 'scope_level', default='')

        quotation_content = f"""
<b>Estimated Project Range:</b> {min_quote} - {max_quote}<br/>
<b>Recommended Quotation:</b> {recommended_quote}<br/>
<b>Scope Level:</b> {scope_level}<br/>
<b>Basis:</b> {quote_basis}
"""
        quotation_para = Paragraph(quotation_content, highlight_box_style)
        quotation_table = Table([[quotation_para]], colWidths=[6.5*inch])
        quotation_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFF9E6')),
            ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#D4AF37')),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))

        story.append(quotation_table)
        story.append(Spacer(1, 0.2*inch))

        cost_factors = safe_get(quotation_range, 'cost_factors', default=[])
        if cost_factors and len(cost_factors) > 0:
            story.append(Paragraph("<b>Key Cost Factors:</b>", body_style))
            for factor in cost_factors:
                story.append(Paragraph(f"• {factor}", body_style))

        story.append(Spacer(1, 0.3*inch))
        story.append(PageBreak())

    # ========== QUOTATION BREAKDOWN SECTION ==========
    if design_data and design_data.get('quotation_breakdown'):
        quotation_breakdown = design_data['quotation_breakdown']
        story.append(Paragraph("Detailed Quotation Breakdown", heading_style))
        story.append(Paragraph("This breakdown provides transparency on how your quotation is calculated based on Singapore 2026 market rates.", body_style))
        story.append(Spacer(1, 0.2*inch))

        cell_style = ParagraphStyle('CellStyle', parent=styles['Normal'], fontSize=9, leading=11)

        breakdown_data = [[
            Paragraph('<b>Category</b>', cell_style),
            Paragraph('<b>Description</b>', cell_style),
            Paragraph('<b>Quantity</b>', cell_style),
            Paragraph('<b>Unit Cost</b>', cell_style),
            Paragraph('<b>Subtotal</b>', cell_style)
        ]]

        categories = [
            ('carpentry', 'Carpentry'),
            ('flooring', 'Flooring'),
            ('painting', 'Painting'),
            ('electrical', 'Electrical Works'),
            ('plumbing', 'Plumbing'),
            ('masonry', 'Masonry & Tiling'),
            ('furniture_fixtures', 'Furniture & Fixtures'),
            ('design_consultation', 'Design & Consultation'),
            ('contingency', 'Contingency Buffer')
        ]

        for key, display_name in categories:
            if key in quotation_breakdown:
                item = quotation_breakdown[key]
                description = safe_get(item, 'description', default=display_name)
                quantity = safe_get(item, 'quantity', default='-')
                unit_cost = safe_get(item, 'unit_cost', default='-')
                subtotal = safe_get(item, 'subtotal', default='TBD')

                breakdown_data.append([
                    Paragraph(f'<b>{display_name}</b>', cell_style),
                    Paragraph(description, cell_style),
                    Paragraph(str(quantity), cell_style),
                    Paragraph(str(unit_cost), cell_style),
                    Paragraph(f'<b>{subtotal}</b>', cell_style)
                ])

        total_quotation = safe_get(quotation_breakdown, 'total_quotation', default='TBD')
        breakdown_data.append([
            Paragraph('<b>TOTAL QUOTATION</b>', cell_style),
            Paragraph('', cell_style),
            Paragraph('', cell_style),
            Paragraph('', cell_style),
            Paragraph(f'<b>{total_quotation}</b>', cell_style)
        ])

        breakdown_table = Table(breakdown_data, colWidths=[1.2*inch, 2*inch, 1*inch, 1.1*inch, 1.2*inch])
        breakdown_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D4AF37')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#FFF9E6')),
            ('LINEABOVE', (0, -1), (-1, -1), 2, colors.HexColor('#D4AF37')),
            ('FONTSIZE', (0, -1), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (3, 1), (4, -1), 'RIGHT'),
        ]))

        story.append(breakdown_table)
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(
            "<b>Note:</b> This quotation is based on standard specifications and Singapore 2026 market rates. "
            "Final costs may vary based on actual product selection, site conditions, and unforeseen circumstances. "
            "The contingency buffer covers unexpected costs and variations.",
            disclaimer_style
        ))
        story.append(PageBreak())

    # ========== FLOOR PLAN IMAGES ==========
    if raw_floor_plan_path and os.path.exists(raw_floor_plan_path):
        try:
            story.append(Paragraph("Original Floor Plan", subheading_style))
            img = Image(raw_floor_plan_path, width=5*inch, height=3.5*inch, kind='proportional')
            story.append(img)
            story.append(Paragraph("Original uploaded floor plan showing the unit layout", caption_style))
            story.append(Spacer(1, 0.3*inch))
        except Exception as e:
            Logger.log(f"[DESIGN DOCS] - ERROR: Could not add raw floor plan image: {e}")

    if segmented_floor_plan_path and os.path.exists(segmented_floor_plan_path):
        try:
            story.append(Paragraph("AI-Segmented Floor Plan", subheading_style))
            seg_img = Image(segmented_floor_plan_path, width=5*inch, height=3.5*inch, kind='proportional')
            story.append(seg_img)
            story.append(Paragraph("AI-processed floor plan with room segmentation and analysis", caption_style))
        except Exception as e:
            Logger.log(f"[DESIGN DOCS] - ERROR: Could not add segmented floor plan image: {e}")

    # ========== AGENT-GENERATED FLOOR PLANS ==========
    valid_agent_floor_plans = [p for p in agent_floor_plan_paths if p and os.path.exists(p)]
    if valid_agent_floor_plans:
        story.append(PageBreak())
        _render_image_grid(
            story=story,
            image_paths=valid_agent_floor_plans,
            heading_style=heading_style,
            subheading_style=subheading_style,
            body_style=body_style,
            caption_style=caption_style,
            section_title="AI-Generated Floor Plans",
            intro_text=(
                "The following floor plan(s) were generated by the PlanPerfect AI Agent based on your "
                "uploaded files and preferences. These provide a spatial overview of proposed layout changes."
            ),
            img_captions=[f"AI Floor Plan {i + 1}" for i in range(len(valid_agent_floor_plans))],
        )

    # ========== AGENT-GENERATED DESIGN IMAGES ==========
    valid_agent_images = [p for p in agent_generated_image_paths if p and os.path.exists(p)]
    if valid_agent_images:
        if not valid_agent_floor_plans:
            story.append(PageBreak())
        _render_image_grid(
            story=story,
            image_paths=valid_agent_images,
            heading_style=heading_style,
            subheading_style=subheading_style,
            body_style=body_style,
            caption_style=caption_style,
            section_title="AI-Generated Design Inspirations",
            intro_text=(
                "The following design image(s) were created by the PlanPerfect AI Agent to visualise "
                "how your space could look after renovation. Use these as a reference when discussing "
                "your project with your interior designer."
            ),
            img_captions=[f"Design Concept {i + 1}" for i in range(len(valid_agent_images))],
        )

    story.append(PageBreak())

    # ========== CLIENT PREFERENCES ==========
    if preferences:
        story.append(Paragraph("Client Preferences", heading_style))

        pref_style = safe_get(preferences, 'style', default='Not specified')
        if pref_style and pref_style not in ['Not specified', 'Not selected']:
            story.append(Paragraph(f"<b>Selected Style:</b> {pref_style}", body_style))
        else:
            story.append(Paragraph(f"<b>Design Style(s):</b> Open to recommendations", body_style))

        colors_list = safe_get(preferences, 'colors', default=[])
        if colors_list and len(colors_list) > 0:
            story.append(Paragraph(f"<b>Suggested Colors:</b> {', '.join(colors_list)}", body_style))
        else:
            story.append(Paragraph(f"<b>Color Palette:</b> Open to designer recommendations", body_style))

        materials_list = safe_get(preferences, 'materials', default=[])
        if materials_list and len(materials_list) > 0:
            story.append(Paragraph(f"<b>Suggested Materials:</b> {', '.join(materials_list)}", body_style))
        else:
            story.append(Paragraph(f"<b>Key Materials:</b> Open to designer recommendations", body_style))

        special_req = safe_get(preferences, 'special_requirements', default='')
        if special_req and special_req.strip():
            story.append(Paragraph(f"<b>Special Requirements:</b> {special_req}", body_style))

        story.append(Spacer(1, 0.3*inch))

    # ========== EXECUTIVE SUMMARY ==========
    if design_data and design_data.get('executive_summary'):
        exec_summary = design_data['executive_summary']
        story.append(Paragraph("1. Executive Summary", heading_style))

        project_overview = safe_get(exec_summary, 'project_overview', default='')
        if project_overview:
            story.append(Paragraph(project_overview, body_style))

        design_philosophy = safe_get(exec_summary, 'design_philosophy', default='')
        if design_philosophy:
            story.append(Paragraph("<b>Design Philosophy:</b>", subheading_style))
            story.append(Paragraph(design_philosophy, body_style))

        key_recs = safe_get(exec_summary, 'key_recommendations', default=[])
        if key_recs and len(key_recs) > 0:
            story.append(Paragraph("<b>Key Recommendations:</b>", subheading_style))
            for rec in key_recs:
                story.append(Paragraph(f"• {rec}", body_style))

        story.append(Spacer(1, 0.3*inch))

    # ========== SPACE ANALYSIS ==========
    if design_data and design_data.get('space_analysis'):
        space_analysis = design_data['space_analysis']
        story.append(Paragraph("2. Space Analysis", heading_style))

        if unit_info:
            unit_sizes = normalize_display_list(safe_get(unit_info, 'unit_sizes', default=[]))
            if unit_sizes:
                story.append(Paragraph(f"<b>Total Area:</b> {', '.join(unit_sizes)}", body_style))
            else:
                story.append(Paragraph(f"<b>Total Area:</b> Not specified", body_style))
        else:
            story.append(Paragraph(f"<b>Total Area:</b> Not specified", body_style))

        room_breakdown = safe_get(space_analysis, 'room_breakdown', default=[])
        if room_breakdown and len(room_breakdown) > 0:
            story.append(Paragraph("<b>Room Breakdown:</b>", subheading_style))
            for room in room_breakdown:
                room_name = safe_get(room, 'room_name', default='Unknown Room')
                analysis = safe_get(room, 'analysis', default='No analysis available')
                story.append(Paragraph(f"<b>{room_name}:</b> {analysis}", body_style))

        story.append(Spacer(1, 0.3*inch))

    # ========== DESIGN CONCEPT ==========
    if design_data and design_data.get('design_concept'):
        design_concept = design_data['design_concept']
        story.append(Paragraph("3. Design Concept", heading_style))

        style_direction = safe_get(design_concept, 'style_direction', default='')
        if style_direction:
            story.append(Paragraph(f"<b>Style Direction:</b> {style_direction}", body_style))

        color_palette = safe_get(design_concept, 'color_palette', default=[])
        if color_palette and len(color_palette) > 0:
            story.append(Paragraph("<b>Color Palette:</b>", subheading_style))
            for color in color_palette:
                story.append(Paragraph(f"• {color}", body_style))

        materials = safe_get(design_concept, 'materials', default=[])
        if materials and len(materials) > 0:
            story.append(Paragraph("<b>Materials:</b>", subheading_style))
            for material in materials:
                story.append(Paragraph(f"• {material}", body_style))

        lighting_strategy = safe_get(design_concept, 'lighting_strategy', default='')
        if lighting_strategy:
            story.append(Paragraph(f"<b>Lighting Strategy:</b> {lighting_strategy}", body_style))

        story.append(PageBreak())

    # ========== ROOM RECOMMENDATIONS ==========
    if design_data and design_data.get('room_recommendations'):
        room_recs = design_data['room_recommendations']
        if room_recs and len(room_recs) > 0:
            story.append(Paragraph("4. Room-Specific Recommendations", heading_style))

            for room in room_recs:
                room_name = safe_get(room, 'room_name', default='Unknown Room')
                story.append(Paragraph(room_name, subheading_style))

                design_approach = safe_get(room, 'design_approach', default='')
                if design_approach:
                    story.append(Paragraph(design_approach, body_style))

                furniture_items = safe_get(room, 'furniture_items', default=[])
                if furniture_items and len(furniture_items) > 0:
                    cell_style = ParagraphStyle('CellStyle', parent=styles['Normal'], fontSize=9, leading=11)

                    furniture_data = [[
                        Paragraph('<b>Item</b>', cell_style),
                        Paragraph('<b>Cost</b>', cell_style),
                        Paragraph('<b>Notes</b>', cell_style)
                    ]]

                    for item in furniture_items:
                        item_name = safe_get(item, 'item', default='Item')
                        item_cost = safe_get(item, 'estimated_cost', default='TBD')
                        item_notes = safe_get(item, 'notes', default='')

                        furniture_data.append([
                            Paragraph(item_name, cell_style),
                            Paragraph(item_cost, cell_style),
                            Paragraph(item_notes, cell_style)
                        ])

                    furniture_table = Table(furniture_data, colWidths=[2*inch, 1*inch, 3.5*inch])
                    furniture_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D4AF37')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('TOPPADDING', (0, 1), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ]))
                    story.append(furniture_table)
                    story.append(Spacer(1, 0.2*inch))

                color_specs = safe_get(room, 'color_specs', default='')
                if color_specs:
                    story.append(Paragraph(f"<b>Color Specifications:</b> {color_specs}", body_style))

                lighting = safe_get(room, 'lighting', default='')
                if lighting:
                    story.append(Paragraph(f"<b>Lighting:</b> {lighting}", body_style))

                story.append(Spacer(1, 0.2*inch))

            story.append(PageBreak())

    # ========== ESTIMATED BUDGET BREAKDOWN ==========
    if design_data and design_data.get('budget_breakdown'):
        budget_breakdown = design_data['budget_breakdown']
        story.append(Paragraph("5. Estimated Budget Breakdown", heading_style))

        total_estimated = safe_get(budget_breakdown, 'total_estimated', default='TBD')
        story.append(Paragraph(f"<b>Total Estimated Cost:</b> {total_estimated}", body_style))

        buffer_amount = safe_get(budget_breakdown, 'buffer_amount', default='TBD')
        story.append(Paragraph(f"<b>Buffer Amount:</b> {buffer_amount}", body_style))

        by_room = safe_get(budget_breakdown, 'by_room', default=[])
        if by_room and len(by_room) > 0:
            cell_style = ParagraphStyle('CellStyle', parent=styles['Normal'], fontSize=9, leading=11)

            budget_data = [[
                Paragraph('<b>Room</b>', cell_style),
                Paragraph('<b>Amount</b>', cell_style),
                Paragraph('<b>Breakdown</b>', cell_style)
            ]]

            for item in by_room:
                room_name = safe_get(item, 'room', default='Room')
                amount = safe_get(item, 'amount', default='TBD')
                breakdown = safe_get(item, 'breakdown', default='')

                budget_data.append([
                    Paragraph(room_name, cell_style),
                    Paragraph(amount, cell_style),
                    Paragraph(breakdown, cell_style)
                ])

            budget_table = Table(budget_data, colWidths=[1.5*inch, 1.2*inch, 3.8*inch])
            budget_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D4AF37')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(budget_table)
            story.append(Spacer(1, 0.2*inch))

        priority_items = safe_get(budget_breakdown, 'priority_items', default=[])
        if priority_items and len(priority_items) > 0:
            story.append(Paragraph("<b>Priority Items (Must-Have):</b>", subheading_style))
            for item in priority_items:
                story.append(Paragraph(f"• {item}", body_style))

        optional_items = safe_get(budget_breakdown, 'optional_items', default=[])
        if optional_items and len(optional_items) > 0:
            story.append(Paragraph("<b>Optional Items (Nice-to-Have):</b>", subheading_style))
            for item in optional_items:
                story.append(Paragraph(f"• {item}", body_style))

        cost_saving_tips = safe_get(budget_breakdown, 'cost_saving_tips', default=[])
        if cost_saving_tips and len(cost_saving_tips) > 0:
            story.append(Paragraph("<b>Cost-Saving Tips:</b>", subheading_style))
            for tip in cost_saving_tips:
                story.append(Paragraph(f"• {tip}", body_style))

        story.append(PageBreak())

    # ========== ESTIMATED IMPLEMENTATION TIMELINE ==========
    if design_data and design_data.get('timeline'):
        timeline = design_data['timeline']
        story.append(Paragraph("6. Estimated Implementation Timeline", heading_style))

        total_duration = safe_get(timeline, 'total_duration', default='')
        if total_duration:
            story.append(Paragraph(f"<b>Estimate Total Project Duration:</b> {total_duration}", body_style))

        phases = safe_get(timeline, 'phases', default=[])
        if phases and len(phases) > 0:
            for phase in phases:
                phase_name = safe_get(phase, 'phase', default='Phase')
                duration = safe_get(phase, 'duration', default='TBD')

                story.append(Paragraph(f"<b>{phase_name}</b> ({duration})", subheading_style))

                budget_allocation = safe_get(phase, 'budget_allocation', default='')
                if budget_allocation:
                    story.append(Paragraph(f"Estimated Budget: {budget_allocation}", body_style))

                tasks = safe_get(phase, 'tasks', default=[])
                if tasks and len(tasks) > 0:
                    story.append(Paragraph("<b>Tasks:</b>", body_style))
                    for task in tasks:
                        story.append(Paragraph(f"• {task}", body_style))

                story.append(Spacer(1, 0.1*inch))

    # ========== NEXT STEPS ==========
    if design_data and design_data.get('next_steps'):
        next_steps = design_data['next_steps']
        if next_steps and len(next_steps) > 0:
            story.append(Paragraph("7. Next Steps", heading_style))
            for step in next_steps:
                story.append(Paragraph(f"• {step}", body_style))

    # ========== MAINTENANCE GUIDE ==========
    if design_data and design_data.get('maintenance_guide'):
        maint = design_data['maintenance_guide']

        daily = safe_get(maint, 'daily', default=[])
        monthly = safe_get(maint, 'monthly', default=[])
        annual = safe_get(maint, 'annual', default=[])

        if (daily and len(daily) > 0) or (monthly and len(monthly) > 0) or (annual and len(annual) > 0):
            story.append(PageBreak())
            story.append(Paragraph("8. Maintenance Guide", heading_style))

        if daily and len(daily) > 0:
            story.append(Paragraph("<b>Daily Maintenance:</b>", subheading_style))
            for item in daily:
                story.append(Paragraph(f"• {item}", body_style))

        if monthly and len(monthly) > 0:
            story.append(Paragraph("<b>Monthly Maintenance:</b>", subheading_style))
            for item in monthly:
                story.append(Paragraph(f"• {item}", body_style))

        if annual and len(annual) > 0:
            story.append(Paragraph("<b>Annual Maintenance:</b>", subheading_style))
            for item in annual:
                story.append(Paragraph(f"• {item}", body_style))

    # ========== CHATBOT CTA ==========
    story.append(Spacer(1, 0.5*inch))

    story.append(HRFlowable(
        width="80%",
        thickness=1,
        color=colors.HexColor('#D4AF37'),
        spaceAfter=20,
        spaceBefore=20,
        hAlign='CENTER'
    ))

    story.append(Paragraph("Want a More Personalized Design Experience?", chatbot_cta_style))

    cta_text_style = ParagraphStyle(
        'CTAText',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#555555'),
        alignment=TA_CENTER,
        spaceAfter=10
    )

    story.append(Paragraph(
        "Chat with our AI Design Assistant for tailored recommendations, instant answers to your design questions, and personalized style guidance!",
        cta_text_style
    ))

    story.append(Paragraph(
        '<b>Visit our chatbot at:</b> www.planperfect.com/chatbot',
        cta_text_style
    ))

    story.append(HRFlowable(
        width="80%",
        thickness=1,
        color=colors.HexColor('#D4AF37'),
        spaceAfter=20,
        spaceBefore=20,
        hAlign='CENTER'
    ))

    story.append(Paragraph(
        "<b>Note:</b> This document is for informational purposes only and does not constitute a binding agreement. Final designs and quotations are subject to change based on further consultations and site assessments.",
        cta_text_style
    ))

    story.append(HRFlowable(
        width="80%",
        thickness=1,
        color=colors.HexColor('#D4AF37'),
        spaceAfter=20,
        spaceBefore=20,
        hAlign='CENTER'
    ))

    story.append(Paragraph(
        "Thank you for choosing PlanPerfect for your interior design needs!",
        cta_text_style
    ))

    story.append(Paragraph(
        "© 2026 PlanPerfect. All rights reserved.",
        cta_text_style
    ))

    doc.build(story)
    return buffer
