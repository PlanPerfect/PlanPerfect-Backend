from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from middleware.auth import _verify_api_key
from datetime import datetime
from groq import Groq
import json
import tempfile
import os
import requests

from Services import DatabaseManager as DM
from Services import FileManager as FM
from Services import RAGManager as RAG
from Services.DesignDocsPdf import generate_pdf
from Services import Logger

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

@router.post("/savePdf/{user_id}")
async def save_generated_pdf(
    user_id: str,
    pdf_file: UploadFile = File(...)
):
    try:
        if not user_id:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "UERROR: One or more required fields are invalid / missing."
                }
            )

        user = DM.peek(["Users", user_id])

        if user is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "UERROR: Please login again."
                }
            )

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_file.filename = f"design_document_{timestamp}.pdf"

        # Upload PDF using FileManager
        upload_result = FM.store_file(
            file=pdf_file,
            subfolder=f"newHomeOwners/{user_id}/documents"
        )

        pdf_url = upload_result["url"]

        if "urls" not in DM.data["Users"][user_id]["Generated Document"] or not isinstance(DM.data["Users"][user_id]["Generated Document"]["urls"], list):
            DM.data["Users"][user_id]["Generated Document"]["urls"] = []

        DM.data["Users"][user_id]["Generated Document"]["urls"].append(pdf_url)

        DM.save()

        return {
            "success": True,
            "result": {
                "pdf_url": pdf_url,
                "filename": pdf_file.filename
            }
        }

    except Exception as e:
        Logger.log(f"[DOCUMENT LLM] - Error saving generated PDF: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"ERROR: Failed to save PDF. {str(e)}"
            }
        )

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
@router.post("/generateDesignDocument/{user_id}")
async def generate_design_document(user_id: str):
    """
    Generates a complete interior design PDF document using LLM.
    Retrieves all necessary data from Firebase RTDB using DatabaseManager.
    """
    tmp_floor_plan_path = None
    tmp_segmented_path = None

    try:
        if not user_id:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "UERROR: One or more required fields are invalid / missing."
                }
            )

        user = DM.peek(["Users", user_id])
        if user is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "UERROR: Please login again."
                }
            )

        # Retrieve user data from DatabaseManager
        user_path = ["Users", user_id, "New Home Owner"]
        user_data = DM.peek(user_path)

        if not user_data:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "ERROR: User data not found."
                }
            )

        # Extract data from database
        preferences_data = user_data.get("Preferences", {})
        budget = preferences_data.get("budget", "Not specified")
        styles = preferences_data.get("Preferred Styles", {}).get("styles", [])

        # Get floor plan URLs
        floor_plan_url = user_data.get("Uploaded Floor Plan", {}).get("url")
        segmented_floor_plan_url = user_data.get("Segmented Floor Plan", {}).get("url")

        # Get unit information
        unit_info_data = user_data.get("Unit Information", {})
        unit_rooms = unit_info_data.get("unit", "Residential Unit")
        unit_type = unit_info_data.get("unitType", "Not specified")
        unit_size = unit_info_data.get("unitSize", "Not specified")

        number_of_rooms = unit_info_data.get("Number Of Rooms", {})

        # Build room counts dictionary
        room_counts = {
            "LIVING": number_of_rooms.get("livingRoom", 0),
            "KITCHEN": number_of_rooms.get("kitchen", 0),
            "BATH": number_of_rooms.get("bathroom", 0),
            "BEDROOM": number_of_rooms.get("bedroom", 0),
            "LEDGE": number_of_rooms.get("ledge", 0),
            "BALCONY": number_of_rooms.get("balcony", 0),
        }

        # Download and save floor plan temporarily if URL exists
        if floor_plan_url:
            try:
                response = requests.get(floor_plan_url)
                response.raise_for_status()

                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(response.content)
                    tmp_floor_plan_path = tmp_file.name
            except Exception as e:
                Logger.log(f"[DOCUMENT LLM] - Error downloading floor plan: {e}")
                tmp_floor_plan_path = None

        # Download and save segmented floor plan temporarily if URL exists
        if segmented_floor_plan_url:
            try:
                response = requests.get(segmented_floor_plan_url)
                response.raise_for_status()

                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(response.content)
                    tmp_segmented_path = tmp_file.name
            except Exception as e:
                Logger.log(f"[DOCUMENT LLM] - Error downloading segmented floor plan: {e}")
                tmp_segmented_path = None

        # Process room counts to create summary
        rooms_summary = []
        for room_type, count in room_counts.items():
            if count and count > 0:
                rooms_summary.append(f"{count} {room_type.title()}")

        property_type = unit_rooms or 'Residential Unit'

        # Retrieve RAG-LLM chatbot history
        rag_history = RAG.get_history(user_id)

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

        # Format styles as comma-separated string
        styles_str = ", ".join(styles) if styles else "Not specified"

        # Prompt to parse in LLM
        prompt = f"""Create a comprehensive interior design document in JSON format.

PROPERTY DETAILS:
- Unit: {property_type}
- Unit Type: {unit_type}
- Unit Size: {unit_size}
- Room Layout: {room_details or 'Standard residential layout'}

CLIENT PREFERENCES (Initial):
- Style: {styles_str}
- Colors: 'Not specified'
- Materials: 'Not specified'
- Special Requirements: 'Not specified'

BUDGET: {budget or 'Not specified'}

SINGAPORE RENOVATION COST REFERENCE (2026):
Home renovation costs in Singapore are projected to rise by 5-7% in 2026. Use these ranges to provide realistic quotations:

HDB BTO/New Units:
- 3-Room: S$36,000 – S$45,000
- 4-Room: S$51,000 – S$65,000
- 5-Room: S$67,000 – S$83,000

HDB Resale Units (20-40% higher due to hacking, rewiring, plumbing):
- 3-Room: S$51,000 – S$62,000
- 4-Room: S$64,000 – S$82,000
- 5-Room: S$84,000 – S$97,000

Condominiums:
- New: S$40,000 – S$75,000
- Resale: S$80,000 – S$105,000

Landed Properties:
- New: S$100,000 – S$250,000+
- Resale: S$300,000 – S$1.5M+

Component Costs (2026):
- Carpentry: S$250 – S$400 per foot run
- Flooring: Vinyl S$4-S$8 psf, Premium tiles/marble S$10-S$47.50 psf
- Electrical: S$1,500-S$4,000 (new), S$4,000-S$15,000 (full rewiring)
- Plumbing: S$1,500 – S$4,000 (bathroom/kitchen updates)

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

10. **QUOTATION REQUIREMENTS**: You MUST provide a detailed quotation breakdown that adds up to the total quotation amount. Base your quotation on the Singapore 2026 renovation cost reference provided above and the property type.

Return JSON with this structure (be detailed and specific):
{{
  "detected_preferences_override": {{
    "style": "detected style from conversation or initial preference (NEVER 'Not specified' - always suggest a style)",
    "colors": ["ALWAYS provide 3-5 specific color suggestions that match the design style, even if user didn't specify. Include color names or hex codes"],
    "materials": ["ALWAYS provide 3-5 specific material suggestions that match the design style, even if user didn't specify"],
    "budget": "detected budget from conversation if mentioned, otherwise use initial budget",
    "special_notes": "any important details from conversation"
  }},
  "quotation_range": {{
    "minimum_quote": "Minimum realistic quote in SGD (e.g., S$51,000) based on property type and Singapore 2026 market rates",
    "maximum_quote": "Maximum realistic quote in SGD (e.g., S$65,000) based on property type and Singapore 2026 market rates",
    "recommended_quote": "Your recommended quote in SGD based on client's budget, requirements, and property type",
    "quote_basis": "Brief explanation of how you arrived at this range (e.g., 'Requiring moderate renovation with quality finishes'). Don't need to mention property type here.",
    "scope_level": "Light/Moderate/Extensive - based on requirements",
    "cost_factors": ["Factor 1 affecting cost with brief explanation", "Factor 2 affecting cost", "Factor 3 affecting cost"]
  }},
  "quotation_breakdown": {{
    "carpentry": {{
      "description": "Custom wardrobes, cabinets, built-in furniture",
      "quantity": "e.g., 40 feet run",
      "unit_cost": "e.g., S$300 per foot run",
      "subtotal": "S$12,000"
    }},
    "flooring": {{
      "description": "Flooring materials and installation",
      "quantity": "e.g., 850 sqft",
      "unit_cost": "e.g., S$6 per sqft",
      "subtotal": "S$5,100"
    }},
    "painting": {{
      "description": "Wall painting and preparation",
      "quantity": "Entire unit",
      "unit_cost": "Flat rate or per sqft",
      "subtotal": "S$3,500"
    }},
    "electrical": {{
      "description": "Electrical works, lighting, wiring",
      "quantity": "Based on scope",
      "unit_cost": "Package or itemized",
      "subtotal": "S$4,500"
    }},
    "plumbing": {{
      "description": "Plumbing works, fixtures",
      "quantity": "Kitchen and bathrooms",
      "unit_cost": "Package or itemized",
      "subtotal": "S$3,200"
    }},
    "masonry": {{
      "description": "Hacking, tiling, masonry work",
      "quantity": "Based on scope",
      "unit_cost": "Package or itemized",
      "subtotal": "S$4,800"
    }},
    "furniture_fixtures": {{
      "description": "Furniture, curtains, accessories",
      "quantity": "As per design",
      "unit_cost": "Itemized or package",
      "subtotal": "S$8,500"
    }},
    "design_consultation": {{
      "description": "Design consultation and project management",
      "quantity": "Full project",
      "unit_cost": "Professional fees",
      "subtotal": "S$3,000"
    }},
    "contingency": {{
      "description": "Buffer for unforeseen costs (typically 10-15%)",
      "quantity": "Percentage of total",
      "unit_cost": "Buffer amount",
      "subtotal": "S$4,460"
    }},
    "total_quotation": "S$49,060 (MUST equal sum of all subtotals above)"
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

CRITICAL: Ensure the quotation_breakdown subtotals add up EXACTLY to the total_quotation amount. Be specific, practical, and ensure all recommendations align with Singapore 2026 market rates and the FINAL preferences (prioritizing conversation history over initial form data) and budget."""

        # Call Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert interior designer with exceptional listening skills and deep knowledge of Singapore's renovation market in 2026. Your specialty is understanding clients' true preferences through conversation and providing accurate, realistic quotations.

CRITICAL ABILITIES:
1. You can detect when a client changes their mind during conversation
2. You prioritize spoken preferences over written forms
3. You notice subtle hints about style preferences, budget concerns, and special needs
4. You create personalized designs that reflect the client's refined preferences
5. You provide detailed, professional recommendations in JSON format
6. You calculate accurate quotations based on Singapore 2026 market rates

IMPORTANT REQUIREMENTS:
- ALWAYS suggest specific colors and materials based on the chosen design style
- Provide professional, style-appropriate recommendations
- ALWAYS provide realistic quotations based on Singapore 2026 renovation costs
- ENSURE quotation breakdown subtotals add up EXACTLY to the total quotation
- Account for scope level (Light, Moderate, or Extensive renovation)
- Example: For "Modern" style → suggest colors like "Crisp White", "Charcoal Gray", "Warm Beige" and materials like "Glass", "Brushed Steel", "Polished Concrete"

Always be specific with measurements, costs, and actionable advice. Ensure the design reflects what the client TRULY wants based on their conversation and that all financial figures are realistic and transparent."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="openai/gpt-oss-120b",
            temperature=0.7,
            max_tokens=5120,
            response_format={"type": "json_object"}
        )

        # Extract response content
        response_text = chat_completion.choices[0].message.content

        # Always clean up encoding issues regardless of model
        response_text_cleaned = (
            response_text
            .replace("\u202f", " ")
            .replace("\u00a0", " ")
            .replace("–", "-")
            .replace("—", "-")
            .replace("'", "'")
            .replace("'", "'")
            .replace("‑", "-")
            .replace("\u2011", "-")
            .replace("×", "x")
            .replace("\u00d7", "x")
            .replace("■", "")
            .replace("\u25a0", "")
        )

        # Extract JSON from code blocks if present
        if "```json" in response_text_cleaned:
            json_str = response_text_cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text_cleaned:
            json_str = response_text_cleaned.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text_cleaned

        # Parse the fully cleaned JSON
        design_data = json.loads(json_str)

        # Build preferences dict for PDF generation
        final_preferences = {
            "style": styles_str,
            "styles": styles
        }

        # Apply conversation overrides
        final_preferences, final_budget = apply_conversation_overrides(
            design_data=design_data,
            base_preferences=final_preferences,
            base_budget=budget
        )

        # Build unit_info dict for PDF generation
        unit_info = {
            "unit_rooms": unit_rooms,
            "unit_types": [unit_type] if unit_type != "Not specified" else [],
            "unit_sizes": [unit_size] if unit_size != "Not specified" else [],
            "room_counts": room_counts
        }

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

        Logger.log(f"[DOCUMENT LLM] - Error generating design document: {str(e)}")

        return JSONResponse(
            status_code=500,
            content={
                "error": f"ERROR: Failed to generate design document. {str(e)}"
            }
        )
