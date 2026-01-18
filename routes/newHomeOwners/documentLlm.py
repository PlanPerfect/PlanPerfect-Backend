from fastapi import APIRouter, File, UploadFile, Depends, Form
from fastapi.responses import StreamingResponse
from middleware.auth import _verify_api_key
from datetime import datetime
from groq import Groq
import json
import tempfile
import os
import base64

from Services import RAGManager as RAG
from Services.DesignDocsPdf import generate_pdf

router = APIRouter(prefix="/newHomeOwners/documentLlm", tags=["LLM PDF Generation"], dependencies=[Depends(_verify_api_key)])

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Helper functions for applying conversation overrides
def _is_placeholder_value(value):
    if not value:
        return True
    
    placeholder_phrases = {
        'not specified',
        'none',
        'open to designer recommendations',
        'open to recommendations',
        ''
    }
    
    return str(value).lower().strip() in placeholder_phrases

def apply_conversation_overrides(design_data, base_preferences, base_budget):
    final_preferences = base_preferences.copy()
    final_budget = base_budget
    
    if 'detected_preferences_override' not in design_data:
        return final_preferences, final_budget
    
    conversation_prefs = design_data['detected_preferences_override']
    
    # Override style if user changed their mind in conversation
    detected_style = conversation_prefs.get('style', '').strip()
    if detected_style and not _is_placeholder_value(detected_style):
        final_preferences['style'] = detected_style
    
    # Always replace LLM suggested colors
    detected_colors = conversation_prefs.get('colors', [])
    if isinstance(detected_colors, list):
        valid_colors = [c for c in detected_colors if not _is_placeholder_value(c)]
        if valid_colors:
            final_preferences['colors'] = valid_colors
    
    # Always replace LLM suggested materials
    detected_materials = conversation_prefs.get('materials', [])
    if isinstance(detected_materials, list):
        valid_materials = [m for m in detected_materials if not _is_placeholder_value(m)]
        if valid_materials:
            final_preferences['materials'] = valid_materials
    
    # Merge special requirements/notes if any
    detected_notes = conversation_prefs.get('special_notes', '').strip()
    if detected_notes and not _is_placeholder_value(detected_notes):
        valid_notes = [n.strip() for n in detected_notes.split(',') if not _is_placeholder_value(n)]
        if valid_notes:
            final_preferences['special_requirements'] = ". ".join(valid_notes)
    
    # Override budget if user mentioned a different budget in conversation
    detected_budget = conversation_prefs.get('budget', '').strip()
    if detected_budget and not _is_placeholder_value(detected_budget):
        final_budget = detected_budget
    
    return final_preferences, final_budget

# Endpoint to generate interior design document
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
        
        # Parse preferences and extraction data from input
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
        
        # Format chat history for LLM if history exists
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
        
        # Prompt to parse in LLM
        prompt = f"""Create a comprehensive interior design document in JSON format.

PROPERTY DETAILS:
- Unit: {property_type}
- Unit Type: {', '.join(unit_types) if unit_types else 'Not specified'}
- Unit Sizes: {', '.join(unit_sizes) if unit_sizes else 'Not specified'}
- Room Layout: {room_details or 'Standard residential layout'}

CLIENT PREFERENCES (Initial):
- Style: {preferences_data.get('style', 'Not specified')}
- Colors: 'Not specified'
- Materials: 'Not specified'
- Special Requirements: 'Not specified'

BUDGET: {budget or 'Not specified'}

{formatted_chat_history}

CRITICAL INSTRUCTIONS FOR HANDLING CONVERSATION HISTORY:
1. **PREFERENCE OVERRIDE**: Carefully analyze the conversation history above. If the user expressed different preferences, styles, or requirements during the conversation that CONTRADICT the initial client preferences, YOU MUST USE THE PREFERENCES FROM THE CONVERSATION instead.
   
2. **STYLE CHANGES**: If the user mentioned they prefer a different design style in the conversation (e.g., "I think boho style suits me better" or "I prefer minimalist" or "I'd like industrial style"), use that style instead of the initial preference.

3. **COLOR PREFERENCES**: If the user discussed specific colors they like or dislike in the conversation, incorporate those preferences and override the initial color preferences if contradictory. If NO colors were specified, YOU MUST suggest appropriate colors based on the chosen design style.

4. **MATERIAL PREFERENCES**: If the user mentioned specific materials they prefer (e.g., "I love wood", "I prefer marble", "I don't like plastic"), prioritize those over the initial material preferences. If NO materials were specified, YOU MUST suggest appropriate materials based on the chosen design style.

5. **BUDGET ADJUSTMENTS**: If the user discussed budget concerns or mentioned a different budget range in the conversation, adjust recommendations accordingly.

6. **SPECIAL REQUIREMENTS**: Extract any additional requirements, concerns, or preferences mentioned in the conversation (e.g., "I have pets", "I need storage space", "I work from home", "I have children"). If NO special requirements were specified, leave it as "Not specified".

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

        # Call Groq API
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
            max_tokens=3072,
            response_format={"type": "json_object"}
        )
        
        # Extract response content
        response_text = chat_completion.choices[0].message.content

        # Clean up encoding issues for GPT models in response text
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
        
        # Parse JSON response into a structured dictionary
        if "```json" in response_text:
            response_text = response_text_cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text_cleaned.split("```")[1].split("```")[0].strip()
        
        design_data = json.loads(response_text)
        
        # Apply conversation overrides 
        final_preferences, final_budget = apply_conversation_overrides(
            design_data=design_data,
            base_preferences=preferences_data,
            base_budget=budget
        )
        
        # Generate PDF
        pdf_buffer = generate_pdf(
            design_data=design_data,
            raw_floor_plan_path=tmp_floor_plan_path,
            segmented_floor_plan_path=tmp_segmented_path,
            preferences=final_preferences,
            budget=final_budget,
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