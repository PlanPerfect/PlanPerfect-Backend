import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, TableStyle, Table, HRFlowable
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from svglib.svglib import svg2rlg
from datetime import datetime
from pathlib import Path

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

def generate_pdf(design_data, raw_floor_plan_path, segmented_floor_plan_path, preferences, budget, unit_info):
    """
    Generate a formatted PDF from the provided design data parameters.
    All sections are optional - if data is missing, the section will be skipped.
    This ensures the PDF can be generated even with incomplete LLM responses.
    """
    # Create PDF document template
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
        fontName='Times-Bold'
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
    
    # Build PDF content
    story = []
    logo_added = False
    
    # ========== LOGO AND TITLE PAGE ==========
    story.append(Spacer(1, 0.3*inch))
    
    # Add logo and text
    BASE_DIR = Path(__file__).resolve().parents[1]
    logo_svg_path = BASE_DIR / "static" / "Logo.svg"
    logo_text_path = BASE_DIR / "static" / "Logo-Text.png"

    # Add SVG logo
    if os.path.exists(logo_svg_path):
        try:
            # Convert SVG to ReportLab drawing
            drawing = svg2rlg(str(logo_svg_path))
            
            # Calculate page center
            page_width = letter[0] - doc.leftMargin - doc.rightMargin
            
            # Logo
            logo_width = 120
            scale_factor = logo_width / drawing.width
            drawing.width = logo_width
            drawing.height = drawing.height * scale_factor
            drawing.scale(scale_factor, scale_factor)
            
            # Center the logo using a table
            logo_table = Table([[drawing]], colWidths=[page_width])
            logo_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(logo_table)
            story.append(Spacer(1, 0.15*inch))
            logo_added = True
            
        except Exception as e:
            print(f"Could not add SVG logo: {e}")
    
    # Add logo text png below the logo
    if os.path.exists(logo_text_path):
        try:
            # Logo text image
            logo_text_img = Image(str(logo_text_path), width=2.5*inch, height=0.6*inch, kind='proportional')
            
            # Center the logo text using a table
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
            print(f"Could not add logo text PNG: {e}")
    
    # Fallback if no logos were added
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
    
    # Property information - all optional
    if unit_info:
        unit_rooms = safe_get(unit_info, 'unit_rooms', default='Residential Unit')
        unit_sizes = safe_get(unit_info, 'unit_sizes', default=[])
        unit_types = safe_get(unit_info, 'unit_types', default=[])
        
        property_info = f"<b>Unit Type:</b> {unit_rooms}"
        story.append(Paragraph(property_info, styles['Normal']))
        
        if unit_types and len(unit_types) > 0:
            story.append(Paragraph(f"<b>Unit Models:</b> {', '.join(set(unit_types))}", styles['Normal']))
        
        if unit_sizes and len(unit_sizes) > 0:
            story.append(Paragraph(f"<b>Unit Sizes:</b> {', '.join(set(unit_sizes))}", styles['Normal']))
    
    if budget:
        story.append(Paragraph(f"<b>Budget:</b> {budget}", styles['Normal']))
    
    story.append(Paragraph(f"<b>Generated On:</b> {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # Add raw floor plan image - optional
    if raw_floor_plan_path and os.path.exists(raw_floor_plan_path):
        try:
            story.append(Paragraph("Original Floor Plan", subheading_style))
            img = Image(raw_floor_plan_path, width=5*inch, height=3.5*inch, kind='proportional')
            story.append(img)
            story.append(Paragraph("Original uploaded floor plan showing the unit layout", caption_style))
            story.append(Spacer(1, 0.3*inch))
        except Exception as e:
            print(f"Could not add raw floor plan image: {e}")
    
    # Add segmented floor plan image - optional
    if segmented_floor_plan_path and os.path.exists(segmented_floor_plan_path):
        try:
            story.append(Paragraph("AI-Segmented Floor Plan", subheading_style))
            seg_img = Image(segmented_floor_plan_path, width=5*inch, height=3.5*inch, kind='proportional')
            story.append(seg_img)
            story.append(Paragraph("AI-processed floor plan with room segmentation and analysis", caption_style))
        except Exception as e:
            print(f"Could not add segmented floor plan image: {e}")
    
    story.append(PageBreak())
    
    # ========== CLIENT PREFERENCES - Optional ==========
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

    # ========== EXECUTIVE SUMMARY - Optional ==========
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
    
    # ========== SPACE ANALYSIS - Optional ==========
    if design_data and design_data.get('space_analysis'):
        space_analysis = design_data['space_analysis']
        story.append(Paragraph("2. Space Analysis", heading_style))
        
        total_area = safe_get(space_analysis, 'total_area', default='N/A')
        story.append(Paragraph(f"<b>Total Area:</b> {total_area}", body_style))
        
        room_breakdown = safe_get(space_analysis, 'room_breakdown', default=[])
        if room_breakdown and len(room_breakdown) > 0:
            story.append(Paragraph("<b>Room Breakdown:</b>", subheading_style))
            for room in room_breakdown:
                room_name = safe_get(room, 'room_name', default='Unknown Room')
                analysis = safe_get(room, 'analysis', default='No analysis available')
                story.append(Paragraph(f"<b>{room_name}:</b> {analysis}", body_style))
        
        story.append(Spacer(1, 0.3*inch))
    
    # ========== DESIGN CONCEPT - Optional ==========
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
    
    # ========== ROOM RECOMMENDATIONS - Optional ==========
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
                
                # Furniture table with notes - optional
                furniture_items = safe_get(room, 'furniture_items', default=[])
                if furniture_items and len(furniture_items) > 0:
                    # Create cell style
                    cell_style = ParagraphStyle(
                        'CellStyle',
                        parent=styles['Normal'],
                        fontSize=9,
                        leading=11
                    )
                    
                    # Header row
                    furniture_data = [[
                        Paragraph('<b>Item</b>', cell_style),
                        Paragraph('<b>Cost</b>', cell_style),
                        Paragraph('<b>Notes</b>', cell_style)
                    ]]
                    
                    # Data rows 
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
    
    # ========== ESTIMATED BUDGET BREAKDOWN - Optional ==========
    if design_data and design_data.get('budget_breakdown'):
        budget_breakdown = design_data['budget_breakdown']
        story.append(Paragraph("5. Estimated Budget Breakdown", heading_style))
        
        total_estimated = safe_get(budget_breakdown, 'total_estimated', default='TBD')
        story.append(Paragraph(f"<b>Total Estimated Cost:</b> {total_estimated}", body_style))
        
        buffer_amount = safe_get(budget_breakdown, 'buffer_amount', default='TBD')
        story.append(Paragraph(f"<b>Buffer Amount:</b> {buffer_amount}", body_style))
        
        by_room = safe_get(budget_breakdown, 'by_room', default=[])
        if by_room and len(by_room) > 0:
            # Create cell style
            cell_style = ParagraphStyle(
                'CellStyle',
                parent=styles['Normal'],
                fontSize=9,
                leading=11
            )
            
            # Header row
            budget_data = [[
                Paragraph('<b>Room</b>', cell_style),
                Paragraph('<b>Amount</b>', cell_style),
                Paragraph('<b>Breakdown</b>', cell_style)
            ]]
            
            # Data rows
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
        
        # Priority items - optional
        priority_items = safe_get(budget_breakdown, 'priority_items', default=[])
        if priority_items and len(priority_items) > 0:
            story.append(Paragraph("<b>Priority Items (Must-Have):</b>", subheading_style))
            for item in priority_items:
                story.append(Paragraph(f"• {item}", body_style))
        
        # Optional items - optional
        optional_items = safe_get(budget_breakdown, 'optional_items', default=[])
        if optional_items and len(optional_items) > 0:
            story.append(Paragraph("<b>Optional Items (Nice-to-Have):</b>", subheading_style))
            for item in optional_items:
                story.append(Paragraph(f"• {item}", body_style))
        
        # Cost-saving tips - optional
        cost_saving_tips = safe_get(budget_breakdown, 'cost_saving_tips', default=[])
        if cost_saving_tips and len(cost_saving_tips) > 0:
            story.append(Paragraph("<b>Cost-Saving Tips:</b>", subheading_style))
            for tip in cost_saving_tips:
                story.append(Paragraph(f"• {tip}", body_style))
        
        story.append(PageBreak())
    
    # ========== ESTIMATED IMPLEMENTATION TIMELINE - Optional ==========
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
    
    # ========== NEXT STEPS - Optional ==========
    if design_data and design_data.get('next_steps'):
        next_steps = design_data['next_steps']
        if next_steps and len(next_steps) > 0:
            story.append(Paragraph("7. Next Steps", heading_style))
            for step in next_steps:
                story.append(Paragraph(f"• {step}", body_style))
    
    # ========== MAINTENANCE GUIDE - Optional ==========
    if design_data and design_data.get('maintenance_guide'):
        maint = design_data['maintenance_guide']
        
        has_content = False
        daily = safe_get(maint, 'daily', default=[])
        monthly = safe_get(maint, 'monthly', default=[])
        annual = safe_get(maint, 'annual', default=[])
        
        if (daily and len(daily) > 0) or (monthly and len(monthly) > 0) or (annual and len(annual) > 0):
            story.append(PageBreak())
            story.append(Paragraph("8. Maintenance Guide", heading_style))
            has_content = True
        
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
    
    # ========== CHATBOT CTA AT END ==========
    story.append(Spacer(1, 0.5*inch))
    
    # Add separator line
    story.append(HRFlowable(
        width="80%",
        thickness=1,
        color=colors.HexColor('#D4AF37'),
        spaceAfter=20,
        spaceBefore=20,
        hAlign='CENTER'
    ))
    
    # Chatbot CTA Information
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
    
    # Build PDF
    doc.build(story)
    
    return buffer