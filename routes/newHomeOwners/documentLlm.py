from fastapi import APIRouter, File, UploadFile, Depends, Form
from fastapi.responses import StreamingResponse
from middleware.auth import _verify_api_key
from groq import Groq
import json
import tempfile
import os
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from svglib.svglib import svg2rlg
from datetime import datetime
from pathlib import Path

from Services import RAGManager as RAG

router = APIRouter(prefix="/newHomeOwners/documentLlm", tags=["LLM PDF Generation"], dependencies=[Depends(_verify_api_key)])

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@router.post("/generateDesignDocument")
async def generate_design_document(
    floor_plan: UploadFile = File(...),
    preferences: str = Form(...),
    budget: str = Form(...),
    extraction_data: str = Form(None),
):
    """
    Generates a complete interior design PDF document using LLM.
    """
    tmp_floor_plan_path = None
    tmp_segmented_path = None
    budget = budget or "Not specified"
    
    try:
        # Save uploaded floor plan temporarily
        if floor_plan:
            suffix = os.path.splitext(floor_plan.filename)[1] if floor_plan.filename else '.png'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                content = await floor_plan.read()
                tmp_file.write(content)
                tmp_floor_plan_path = tmp_file.name
        
        # Parse input data
        preferences_data = json.loads(preferences) if preferences else {}
        extraction_data_parsed = json.loads(extraction_data) if extraction_data else {}
        
        # Extract segmented floor plan (base64 encoded image)
        segmented_image_base64 = extraction_data_parsed.get('segmentedImage', None)
        
        # Process segmented image if it exists
        if segmented_image_base64:
            try:
                # Remove the data URI prefix if present (e.g., "data:image/png;base64,")
                if ',' in segmented_image_base64:
                    segmented_image_base64 = segmented_image_base64.split(',')[1]
                
                # Decode base64 and save to temporary file
                segmented_image_data = base64.b64decode(segmented_image_base64)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_seg_file:
                    tmp_seg_file.write(segmented_image_data)
                    tmp_segmented_path = tmp_seg_file.name
            except Exception as e:
                print(f"Error processing segmented image: {e}")
                tmp_segmented_path = None

        # Extract room information
        unit_info = extraction_data_parsed.get('unitInfo', {})
        
        # Process room counts
        room_counts = unit_info.get('room_counts', {})
        rooms_summary = []
        for room_type, count in room_counts.items():
            if count > 0:
                rooms_summary.append(f"{count} {room_type.title()}")
        
        property_type = unit_info.get('unit_rooms', 'Residential Unit')
        unit_types = unit_info.get('unit_types', [])
        unit_sizes = unit_info.get('unit_sizes', [])
        
        # Retrieve RAG-LLM chatbot history
        rag_history = RAG.get_history()
        
        # Format chat history for LLM
        formatted_chat_history = ""
        if rag_history and len(rag_history) > 0:
            formatted_chat_history = "CONVERSATION HISTORY:\n"
            for msg in rag_history:
                role = msg.get('role', 'unknown').upper()
                content = msg.get('content', '')
                formatted_chat_history += f"{role}: {content}\n"
            formatted_chat_history += "\n"
        
        # Build detailed room information for prompt
        room_details = ""
        if rooms_summary:
            room_details = "Identified Rooms:\n"
            room_details += ", ".join(rooms_summary)
        
        # Enhanced prompt with intelligent context handling
        prompt = f"""Create a comprehensive interior design document in JSON format.

PROPERTY DETAILS:
- Unit: {property_type}
- Unit Type: {', '.join(unit_types) if unit_types else 'Not specified'}
- Unit Sizes: {', '.join(unit_sizes) if unit_sizes else 'Not specified'}
- Room Layout: {room_details or 'Standard residential layout'}

CLIENT PREFERENCES (Initial):
- Style: {preferences_data.get('style', 'Not specified')}
- Colors: {', '.join(preferences_data.get('colors', [])) or 'Not specified'}
- Materials: {', '.join(preferences_data.get('materials', [])) or 'Not specified'}
- Special Requirements: {preferences_data.get('special_requirements', 'None')}

BUDGET: {budget or 'Not specified'}

{formatted_chat_history}

CRITICAL INSTRUCTIONS FOR HANDLING CONVERSATION HISTORY:
1. **PREFERENCE OVERRIDE**: Carefully analyze the conversation history above. If the user expressed different preferences, styles, or requirements during the conversation that CONTRADICT the initial client preferences, YOU MUST USE THE PREFERENCES FROM THE CONVERSATION instead.
   
2. **STYLE CHANGES**: If the user mentioned they prefer a different design style in the conversation (e.g., "I think boho style suits me better" or "I prefer minimalist" or "I'd like industrial style"), use that style instead of the initial preference.

3. **COLOR PREFERENCES**: If the user discussed specific colors they like or dislike in the conversation, incorporate those preferences and override the initial color preferences if contradictory. If NO colors were specified, YOU MUST suggest appropriate colors based on the chosen design style.

4. **MATERIAL PREFERENCES**: If the user mentioned specific materials they prefer (e.g., "I love wood", "I prefer marble", "I don't like plastic"), prioritize those over the initial material preferences. If NO materials were specified, YOU MUST suggest appropriate materials based on the chosen design style.

5. **BUDGET ADJUSTMENTS**: If the user discussed budget concerns or mentioned a different budget range in the conversation, adjust recommendations accordingly.

6. **SPECIAL REQUIREMENTS**: Extract any additional requirements, concerns, or preferences mentioned in the conversation (e.g., "I have pets", "I need storage space", "I work from home", "I have children").

7. **NAME PERSONALIZATION**: If the user provided their name in the conversation, acknowledge it naturally in the document.

8. **PRIORITIZATION**: Conversation history takes PRECEDENCE over initial form preferences when there are conflicts. The conversation reveals the user's true, refined preferences.

9. **ALWAYS PROVIDE SUGGESTIONS**: Even if the user didn't specify colors or materials, you MUST provide professional recommendations based on the chosen design style. Never leave these fields as "Not specified" or "Open to recommendations".

Return JSON with this structure (be detailed and specific):
{{
  "detected_preferences_override": {{
    "style": "detected style from conversation or initial preference (NEVER 'Not specified' - always suggest a style)",
    "colors": ["ALWAYS provide 3-5 specific color suggestions that match the design style, even if user didn't specify. Include color names or hex codes"],
    "materials": ["ALWAYS provide 3-5 specific material suggestions that match the design style, even if user didn't specify"],
    "budget": "detected budget from conversation if mentioned, otherwise use initial budget",
    "special_notes": "any important details from conversation"
  }},
  "executive_summary": {{
    "project_overview": "4-5 sentence comprehensive overview including property type, size, and goals. If user provided name, mention it naturally.",
    "design_philosophy": "3-4 sentences explaining the design approach based on FINAL preferences (from conversation history if applicable)",
    "key_recommendations": ["rec1", "rec2", "rec3", "rec4", "rec5"]
  }},
  "space_analysis": {{
    "total_area": "Unit Sizes from Property Details",
    "room_breakdown": [
      {{"room_name": "Living Room", "analysis": "2-3 sentences on design potential and recommendations"}},
      [Skip "Ledges" for space analysis]
    ]
  }},
  "design_concept": {{
    "style_direction": "2-3 sentences on overall aesthetic based on FINAL preferences",
    "color_palette": ["Primary: color1 (usage)", "Secondary: color2 (usage)", "Accent: color3 (usage)"],
    "materials": ["material1 (where to use)", "material2 (where to use)", "material3 (where to use)"],
    "lighting_strategy": "2-3 sentences on layered lighting approach"
  }},
  "room_recommendations": [
    {{
      "room_name": "Room Name",
      "design_approach": "3-4 sentences on design strategy for this specific room based on conversation insights",
      "furniture_items": [
        {{"item": "Specific item name", "estimated_cost": "$XXX", "notes": "Why this piece fits the space and user preferences"}},
        {{"item": "Item 2", "estimated_cost": "$XXX", "notes": "Justification"}}
      ],
      "color_specs": "2 sentences on color application in this room",
      "lighting": "2 sentences on lighting plan for this room"
    }}
  ],
  "budget_breakdown": {{
    "total_estimated": {budget or "Total amount within moderate constraint"},
    "buffer_amount": "$XXX buffer for unexpected costs",
    "by_room": [{{"room": "Room Name", "amount": "$XXXX", "breakdown": "brief breakdown"}}],
    "priority_items": ["Must-have item 1 ($cost)", "Must-have item 2 ($cost)"],
    "optional_items": ["Nice-to-have 1 ($cost)", "Nice-to-have 2 ($cost)"],
    "cost_saving_tips": ["Tip 1", "Tip 2", "Tip 3"]
  }},
  "timeline": {{
    "total_duration": "X weeks/months",
    "phases": [
      {{
        "phase": "Phase name",
        "duration": "X weeks",
        "tasks": ["Detailed task 1", "Detailed task 2", "Detailed task 3"],
        "budget_allocation": "$XXXX"
      }}
    ]
  }},
  "next_steps": [
    "Detailed step 1 with specific actions",
    "Detailed step 2 with timeline",
    "Detailed step 3 with contacts/resources"
  ],
  "maintenance_guide": {{
    "daily": ["Daily maintenance tip 1", "Daily tip 2"],
    "monthly": ["Monthly maintenance task 1", "Monthly task 2"],
    "annual": ["Annual maintenance 1", "Annual maintenance 2"]
  }}
}}

Be specific, practical, and ensure all recommendations align with the FINAL preferences (prioritizing conversation history over initial form data) and budget."""

        # Call Groq API with enhanced context awareness
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert interior designer with exceptional listening skills. Your specialty is understanding clients' true preferences through conversation.

CRITICAL ABILITIES:
1. You can detect when a client changes their mind during conversation
2. You prioritize spoken preferences over written forms
3. You notice subtle hints about style preferences, budget concerns, and special needs
4. You create personalized designs that reflect the client's refined preferences
5. You provide detailed, professional recommendations in JSON format

IMPORTANT REQUIREMENT:
- ALWAYS suggest specific colors and materials, even if the client didn't specify them
- Base your color and material suggestions on the chosen design style
- Provide professional, style-appropriate recommendations
- Example: For "Modern" style → suggest colors like "Crisp White", "Charcoal Gray", "Warm Beige" and materials like "Glass", "Brushed Steel", "Polished Concrete"
- Example: For "Boho" style → suggest colors like "Terracotta", "Sage Green", "Mustard Yellow" and materials like "Rattan", "Macrame", "Reclaimed Wood"

Always be specific with measurements, costs, and actionable advice. Ensure the design reflects what the client TRULY wants based on their conversation."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=4096,
            response_format={"type": "json_object"}
        )
        
        # Extract response
        response_text = chat_completion.choices[0].message.content

        if chat_completion.model.startswith("gpt"):
            response_text_cleaned = (
                response_text
                .replace("\u202f", " ")
                .replace("–", "-")
                .replace("—", "-")
                .replace("“", '"')
                .replace("”", '"')
                .replace("‘", "'")
                .replace("’", "'")
            )
        else:
            response_text_cleaned = response_text
        
        # Parse JSON response
        if "```json" in response_text:
            response_text = response_text_cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text_cleaned.split("```")[1].split("```")[0].strip()
        
        design_data = json.loads(response_text)
        
        # Override preferences with detected preferences from conversation
        final_preferences = preferences_data.copy()
        final_budget = budget
        
        if 'detected_preferences_override' in design_data:
            overrides = design_data['detected_preferences_override']
            
            # Override style (if detected and different from original)
            if overrides.get('style') and overrides['style'].lower() not in ['not specified', 'none', '']:
                final_preferences['style'] = overrides['style']
            
            # Override colors (merge or replace based on conversation)
            if overrides.get('colors'):
                detected_colors = overrides['colors']
                if isinstance(detected_colors, list) and len(detected_colors) > 0:
                    # Filter out generic placeholders
                    valid_colors = [c for c in detected_colors if c.lower() not in ['not specified', 'none', '']]
                    if valid_colors:
                        final_preferences['colors'] = valid_colors
            
            # Override materials (merge or replace based on conversation)
            if overrides.get('materials'):
                detected_materials = overrides['materials']
                if isinstance(detected_materials, list) and len(detected_materials) > 0:
                    # Filter out generic placeholders
                    valid_materials = [m for m in detected_materials if m.lower() not in ['not specified', 'none', 'open to designer recommendations', '']]
                    if valid_materials:
                        final_preferences['materials'] = valid_materials
            
            # Override or append special requirements
            if overrides.get('special_notes') and overrides['special_notes'].strip():
                detected_notes = overrides['special_notes'].strip()
                original_requirements = final_preferences.get('special_requirements', '').strip()
                
                # If original requirements exist and are different, combine them
                if original_requirements and original_requirements.lower() != 'none':
                    # Check if the detected notes are substantially different
                    if detected_notes.lower() not in original_requirements.lower():
                        final_preferences['special_requirements'] = f"{original_requirements}. {detected_notes}"
                    else:
                        final_preferences['special_requirements'] = detected_notes
                else:
                    final_preferences['special_requirements'] = detected_notes
            
            # Override budget if mentioned in conversation
            if overrides.get('budget') and overrides['budget'].lower() not in ['not specified', 'none', '']:
                final_budget = overrides['budget']
        
        # Generate PDF
        pdf_buffer = generate_pdf(
            design_data=design_data,
            raw_floor_plan_path=tmp_floor_plan_path,
            segmented_floor_plan_path=tmp_segmented_path,
            preferences=final_preferences,
            budget=final_budget or 'Not specified',
            unit_info=unit_info
        )
        
        # Clean up temporary files
        if tmp_floor_plan_path and os.path.exists(tmp_floor_plan_path):
            os.unlink(tmp_floor_plan_path)
        if tmp_segmented_path and os.path.exists(tmp_segmented_path):
            os.unlink(tmp_segmented_path)
        
        # Return PDF as streaming response
        pdf_buffer.seek(0)
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=interior_design_documentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )
        
    except Exception as e:
        # Clean up temporary files if they exist
        if tmp_floor_plan_path and os.path.exists(tmp_floor_plan_path):
            try:
                os.unlink(tmp_floor_plan_path)
            except:
                pass
        if tmp_segmented_path and os.path.exists(tmp_segmented_path):
            try:
                os.unlink(tmp_segmented_path)
            except:
                pass
        
        print(f"Error generating design document: {str(e)}")
        raise Exception(f"Failed to generate design document: {str(e)}")

def generate_pdf(design_data, raw_floor_plan_path, segmented_floor_plan_path, preferences, budget, unit_info):
    """
    Generate a formatted PDF from the design data parameters generated by LLM.
    """
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
    
    # ========== LOGO AND TITLE PAGE ==========
    story.append(Spacer(1, 0.3*inch))
    
    # Add logo and text
    BASE_DIR = Path(__file__).resolve().parents[2]
    logo_svg_path = BASE_DIR / "static" / "Logo.svg"
    logo_text_path = BASE_DIR / "static" / "Logo-Text.png"

    # Add SVG logo
    if os.path.exists(logo_svg_path):
        try:
            # Convert SVG to ReportLab drawing
            drawing = svg2rlg(str(logo_svg_path))
            
            # Calculate page center
            page_width = letter[0] - doc.leftMargin - doc.rightMargin
            
            # Scale logo to appropriate size (e.g., 120 points width)
            logo_width = 120
            scale_factor = logo_width / drawing.width
            drawing.width = logo_width
            drawing.height = drawing.height * scale_factor
            drawing.scale(scale_factor, scale_factor)
            
            # Center the drawing by wrapping in a table
            from reportlab.platypus import Table
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
    
    # Add PNG logo text below the logo
    if os.path.exists(logo_text_path):
        try:
            # Create image with appropriate width (e.g., 200 points)
            logo_text_img = Image(str(logo_text_path), width=2.5*inch, height=0.6*inch, kind='proportional')
            
            # Center the logo text using a table
            from reportlab.platypus import Table
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
    
    # Property information
    unit_rooms = unit_info.get('unit_rooms', 'Residential Unit')
    unit_sizes = unit_info.get('unit_sizes', [])
    unit_types = unit_info.get('unit_types', [])
    
    property_info = f"<b>Unit Type:</b> {unit_rooms}"
    story.append(Paragraph(property_info, styles['Normal']))
    
    if unit_types:
        story.append(Paragraph(f"<b>Unit Models:</b> {', '.join(set(unit_types))}", styles['Normal']))
    
    if unit_sizes:
        story.append(Paragraph(f"<b>Unit Sizes:</b> {', '.join(set(unit_sizes))}", styles['Normal']))
    
    story.append(Paragraph(f"<b>Budget:</b> {budget}", styles['Normal']))
    story.append(Paragraph(f"<b>Generated On:</b> {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # Add raw floor plan image
    if raw_floor_plan_path and os.path.exists(raw_floor_plan_path):
        try:
            story.append(Paragraph("Original Floor Plan", subheading_style))
            img = Image(raw_floor_plan_path, width=5*inch, height=3.5*inch, kind='proportional')
            story.append(img)
            story.append(Paragraph("Original uploaded floor plan showing the unit layout", caption_style))
            story.append(Spacer(1, 0.3*inch))
        except Exception as e:
            print(f"Could not add raw floor plan image: {e}")
    
    # Add segmented floor plan image
    if segmented_floor_plan_path and os.path.exists(segmented_floor_plan_path):
        try:
            story.append(Paragraph("AI-Segmented Floor Plan", subheading_style))
            seg_img = Image(segmented_floor_plan_path, width=5*inch, height=3.5*inch, kind='proportional')
            story.append(seg_img)
            story.append(Paragraph("AI-processed floor plan with room segmentation and analysis", caption_style))
        except Exception as e:
            print(f"Could not add segmented floor plan image: {e}")
    
    story.append(PageBreak())
    
    # ========== CLIENT PREFERENCES ==========
    story.append(Paragraph("Client Preferences", heading_style))
    
    pref_style = preferences.get('style', 'Not specified')
    if pref_style and pref_style != 'Not selected':
        story.append(Paragraph(f"<b>Selected Style:</b> {pref_style}", body_style))
    else:
        story.append(Paragraph(f"<b>Design Style(s):</b> Open to recommendations", body_style))
    
    if preferences.get('colors') and len(preferences['colors']) > 0:
        story.append(Paragraph(f"<b>Suggested Colors:</b> {', '.join(preferences['colors'])}", body_style))
    else:
        story.append(Paragraph(f"<b>Color Palette:</b> Open to designer recommendations", body_style))
    
    if preferences.get('materials') and len(preferences['materials']) > 0:
        story.append(Paragraph(f"<b>Suggested Materials:</b> {', '.join(preferences['materials'])}", body_style))
    else:
        story.append(Paragraph(f"<b>Key Materials:</b> Open to designer recommendations", body_style))
    
    if preferences.get('special_requirements') and preferences['special_requirements'].strip():
        story.append(Paragraph(f"<b>Special Requirements:</b> {preferences['special_requirements']}", body_style))
    
    story.append(Spacer(1, 0.3*inch))

    # ========== EXECUTIVE SUMMARY ==========
    story.append(Paragraph("1. Executive Summary", heading_style))
    story.append(Paragraph(design_data['executive_summary']['project_overview'], body_style))
    story.append(Paragraph("<b>Design Philosophy:</b>", subheading_style))
    story.append(Paragraph(design_data['executive_summary']['design_philosophy'], body_style))
    story.append(Paragraph("<b>Key Recommendations:</b>", subheading_style))
    for rec in design_data['executive_summary']['key_recommendations']:
        story.append(Paragraph(f"• {rec}", body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ========== SPACE ANALYSIS ==========
    story.append(Paragraph("2. Space Analysis", heading_style))
    story.append(Paragraph(f"<b>Total Area:</b> {design_data['space_analysis'].get('total_area', 'N/A')}", body_style))
    story.append(Paragraph("<b>Room Breakdown:</b>", subheading_style))
    for room in design_data['space_analysis']['room_breakdown']:
        story.append(Paragraph(f"<b>{room['room_name']}:</b> {room['analysis']}", body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ========== DESIGN CONCEPT ==========
    story.append(Paragraph("3. Design Concept", heading_style))
    story.append(Paragraph(f"<b>Style Direction:</b> {design_data['design_concept']['style_direction']}", body_style))
    
    story.append(Paragraph("<b>Color Palette:</b>", subheading_style))
    for color in design_data['design_concept']['color_palette']:
        story.append(Paragraph(f"• {color}", body_style))
    
    story.append(Paragraph("<b>Materials:</b>", subheading_style))
    for material in design_data['design_concept']['materials']:
        story.append(Paragraph(f"• {material}", body_style))
    
    story.append(Paragraph(f"<b>Lighting Strategy:</b> {design_data['design_concept']['lighting_strategy']}", body_style))
    story.append(PageBreak())
    
    # ========== ROOM RECOMMENDATIONS ==========
    story.append(Paragraph("4. Room-Specific Recommendations", heading_style))
    for room in design_data['room_recommendations']:
        story.append(Paragraph(room['room_name'], subheading_style))
        story.append(Paragraph(room['design_approach'], body_style))
        
        # Furniture table with notes
        if room.get('furniture_items'):
            # Create cell style for wrapping text
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
            
            # Data rows with Paragraphs for text wrapping
            for item in room['furniture_items']:
                furniture_data.append([
                    Paragraph(item['item'], cell_style),
                    Paragraph(item['estimated_cost'], cell_style),
                    Paragraph(item.get('notes', ''), cell_style)
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
        
        story.append(Paragraph(f"<b>Color Specifications:</b> {room['color_specs']}", body_style))
        story.append(Paragraph(f"<b>Lighting:</b> {room['lighting']}", body_style))
        story.append(Spacer(1, 0.2*inch))
    
    story.append(PageBreak())
    
    # ========== BUDGET BREAKDOWN ==========
    story.append(Paragraph("5. Estimated Budget Breakdown", heading_style))
    story.append(Paragraph(f"<b>Total Estimated Cost:</b> {design_data['budget_breakdown']['total_estimated']}", body_style))
    story.append(Paragraph(f"<b>Buffer Amount:</b> {design_data['budget_breakdown']['buffer_amount']}", body_style))
    
    # Create cell style for wrapping text
    cell_style = ParagraphStyle(
        'CellStyle',
        parent=styles['Normal'],
        fontSize=9,
        leading=11
    )
    
    # By room table with wrapped text
    budget_data = [[
        Paragraph('<b>Room</b>', cell_style),
        Paragraph('<b>Amount</b>', cell_style),
        Paragraph('<b>Breakdown</b>', cell_style)
    ]]
    
    for item in design_data['budget_breakdown']['by_room']:
        budget_data.append([
            Paragraph(item['room'], cell_style),
            Paragraph(item['amount'], cell_style),
            Paragraph(item.get('breakdown', ''), cell_style)
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
    
    story.append(Paragraph("<b>Priority Items (Must-Have):</b>", subheading_style))
    for item in design_data['budget_breakdown']['priority_items']:
        story.append(Paragraph(f"• {item}", body_style))
    
    story.append(Paragraph("<b>Optional Items (Nice-to-Have):</b>", subheading_style))
    for item in design_data['budget_breakdown'].get('optional_items', []):
        story.append(Paragraph(f"• {item}", body_style))
    
    if design_data['budget_breakdown'].get('cost_saving_tips'):
        story.append(Paragraph("<b>Cost-Saving Tips:</b>", subheading_style))
        for tip in design_data['budget_breakdown']['cost_saving_tips']:
            story.append(Paragraph(f"• {tip}", body_style))
    
    story.append(PageBreak())
    
    # ========== TIMELINE ==========
    story.append(Paragraph("6. Estimated Implementation Timeline", heading_style))
    if design_data['timeline'].get('total_duration'):
        story.append(Paragraph(f"<b>Estimate Total Project Duration:</b> {design_data['timeline']['total_duration']}", body_style))
    
    for phase in design_data['timeline']['phases']:
        story.append(Paragraph(f"<b>{phase['phase']}</b> ({phase['duration']})", subheading_style))
        if phase.get('budget_allocation'):
            story.append(Paragraph(f"Estimated Budget: {phase['budget_allocation']}", body_style))
        story.append(Paragraph("<b>Tasks:</b>", body_style))
        for task in phase['tasks']:
            story.append(Paragraph(f"• {task}", body_style))
        story.append(Spacer(1, 0.1*inch))
    
    # ========== NEXT STEPS ==========
    story.append(Paragraph("7. Next Steps", heading_style))
    for step in design_data['next_steps']:
        story.append(Paragraph(f"• {step}", body_style))
    
    # ========== MAINTENANCE GUIDE ==========
    if design_data.get('maintenance_guide'):
        story.append(PageBreak())
        story.append(Paragraph("8. Maintenance Guide", heading_style))
        
        maint = design_data['maintenance_guide']
        
        if maint.get('daily'):
            story.append(Paragraph("<b>Daily Maintenance:</b>", subheading_style))
            for item in maint['daily']:
                story.append(Paragraph(f"• {item}", body_style))
        
        if maint.get('monthly'):
            story.append(Paragraph("<b>Monthly Maintenance:</b>", subheading_style))
            for item in maint['monthly']:
                story.append(Paragraph(f"• {item}", body_style))
        
        if maint.get('annual'):
            story.append(Paragraph("<b>Annual Maintenance:</b>", subheading_style))
            for item in maint['annual']:
                story.append(Paragraph(f"• {item}", body_style))
    
    # ========== CHATBOT CTA AT END ==========
    story.append(Spacer(1, 0.5*inch))
    
    # Add separator line
    from reportlab.platypus import HRFlowable
    story.append(HRFlowable(
        width="80%",
        thickness=1,
        color=colors.HexColor('#D4AF37'),
        spaceAfter=20,
        spaceBefore=20,
        hAlign='CENTER'
    ))
    
    # Chatbot CTA
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