from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from middleware.auth import _verify_api_key
from datetime import datetime
from groq import Groq
import json
import re
import tempfile
import os
import requests

from Services import DatabaseManager as DM
from Services import FileManager as FM
from Services import RAGManager as RAG
from Services.ExistingHomeOwnerDesignDocsPdf import generate_pdf
from Services import Logger

router = APIRouter(prefix="/designDocument/existingHomeOwnerDocumentLlm", tags=["LLM PDF Generation - Existing Home Owner"], dependencies=[Depends(_verify_api_key)])

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
groq_alt_client = Groq(api_key=os.getenv("GROQ_ALT_API_KEY"))

# Unicode cleaning
UNICODE_REPLACEMENTS = str.maketrans({
    "\u202f": " ",
    "\u00a0": " ",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2011": "-",
    "\u2012": "-",
    "\u00d7": "x",
    "\u25a0": "",
    "\u00ab": '"',
    "\u00bb": '"',
})

def _clean_text(text: str) -> str:
    return text.translate(UNICODE_REPLACEMENTS)

# JSON extraction & repair 
def _extract_json_str(raw: str) -> str:
    raw = raw.strip()
    if "```json" in raw:
        return raw.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in raw:
        return raw.split("```", 1)[1].split("```", 1)[0].strip()
    return raw

def _attempt_repair(bad_json: str) -> dict:
    attempts = [bad_json]

    # Fix unquoted string values
    repaired = re.sub(
        r':\s*([A-Za-z][^",\}\]\n]{0,120})"',
        lambda m: ': "' + m.group(1).replace('"', "'") + '"',
        bad_json,
    )
    attempts.append(repaired)

    # Remove trailing commas before } or ]
    no_trailing = re.sub(r',\s*([}\]])', r'\1', bad_json)
    attempts.append(no_trailing)

    # Both repairs combined
    both = re.sub(r',\s*([}\]])', r'\1', repaired)
    attempts.append(both)

    for attempt in attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue

    raise ValueError("All JSON repair attempts failed.")

def _parse_llm_response(response_text: str) -> dict:
    cleaned = _clean_text(response_text)
    json_str = _extract_json_str(cleaned)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        Logger.log("[EXISTING DOCUMENT LLM] - ERROR: Initial JSON parse failed, attempting repair...")
        return _attempt_repair(json_str)

def _call_groq(system_prompt: str, user_prompt: str) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    params = dict(
        messages=messages,
        model="openai/gpt-oss-120b",
        temperature=0.7,
        max_tokens=5120,
        response_format={"type": "json_object"},
    )

    # Primary key attempt
    try:
        completion = groq_client.chat.completions.create(**params)
        return _parse_llm_response(completion.choices[0].message.content)
    except Exception as primary_err:
        Logger.log(f"[DOCUMENT LLM] - WARNING: Primary Groq key failed ({primary_err}), retrying with alt key...")

    # Alt key attempt 
    try:
        completion = groq_alt_client.chat.completions.create(**params)
        return _parse_llm_response(completion.choices[0].message.content)
    except Exception as groq_err:
        err_str = str(groq_err)
        failed_gen = None
        try:
            match = re.search(r"'failed_generation':\s*'(.*?)'(?:\s*\})", err_str, re.DOTALL)
            if match:
                failed_gen = match.group(1).encode("utf-8").decode("unicode_escape")
            else:
                body_match = re.search(r"\{.*\}", err_str, re.DOTALL)
                if body_match:
                    body = json.loads(body_match.group(0).replace("'", '"'))
                    failed_gen = body.get("error", {}).get("failed_generation")
        except Exception:
            pass

        if failed_gen:
            Logger.log("[DOCUMENT LLM] - ERROR: Groq json_validate_failed - attempting to parse failed_generation...")
            try:
                return _parse_llm_response(failed_gen)
            except Exception as repair_err:
                Logger.log(f"[DOCUMENT LLM] - ERROR: Repair also failed: {repair_err}")

        raise groq_err

# Placeholder helper 
def _is_placeholder_value(value):
    if not value:
        return True
    placeholder_phrases = {
        'not specified', 'none',
        'open to designer recommendations',
        'open to recommendations', '',
    }
    return str(value).lower().strip() in placeholder_phrases

# Conversation override helper
def apply_conversation_overrides(design_data, base_preferences, base_budget):
    final_preferences = base_preferences.copy()
    final_budget = base_budget

    if 'detected_preferences_override' not in design_data:
        return final_preferences, final_budget

    conversation_prefs = design_data['detected_preferences_override']

    detected_style = conversation_prefs.get('style', '').strip()
    if detected_style and not _is_placeholder_value(detected_style):
        final_preferences['style'] = detected_style

    detected_colors = conversation_prefs.get('colors', [])
    if isinstance(detected_colors, list):
        valid_colors = [c for c in detected_colors if not _is_placeholder_value(c)]
        if valid_colors:
            final_preferences['colors'] = valid_colors

    detected_materials = conversation_prefs.get('materials', [])
    if isinstance(detected_materials, list):
        valid_materials = [m for m in detected_materials if not _is_placeholder_value(m)]
        if valid_materials:
            final_preferences['materials'] = valid_materials

    detected_notes = conversation_prefs.get('special_notes', '').strip()
    if detected_notes and not _is_placeholder_value(detected_notes):
        valid_notes = [n.strip() for n in detected_notes.split(',') if not _is_placeholder_value(n)]
        if valid_notes:
            final_preferences['special_requirements'] = ". ".join(valid_notes)

    detected_budget = conversation_prefs.get('budget', '').strip()
    if detected_budget and not _is_placeholder_value(detected_budget):
        final_budget = detected_budget

    return final_preferences, final_budget

# Download a list of image URLs to temp files 
def _download_image_list(urls: list, suffix: str = '.jpg') -> list:
    """Download a list of image URLs, returning a list of local temp file paths."""
    paths = []
    for url in urls:
        if not url:
            continue
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(r.content)
                paths.append(f.name)
        except Exception as e:
            Logger.log(f"[EXISTING DOCUMENT LLM] - ERROR: Could not download image {url}: {e}")
    return paths

# Save PDF endpoint 
@router.post("/savePdf/{user_id}")
async def save_generated_pdf(user_id: str, pdf_file: UploadFile = File(...)):
    try:
        if not user_id:
            return JSONResponse(status_code=400, content={"error": "UERROR: One or more required fields are invalid / missing."})

        user = DM.peek(["Users", user_id])
        if user is None:
            return JSONResponse(status_code=404, content={"error": "UERROR: Please login again."})

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_file.filename = f"design_document_existing_{timestamp}.pdf"

        upload_result = FM.store_file(file=pdf_file, subfolder=f"existingHomeOwners/{user_id}/documents")
        pdf_url = upload_result["url"]

        if "Generated Document" not in DM.data["Users"][user_id]:
            DM.data["Users"][user_id]["Generated Document"] = {}

        if not isinstance(DM.data["Users"][user_id]["Generated Document"].get("urls"), list):
            DM.data["Users"][user_id]["Generated Document"]["urls"] = []

        DM.data["Users"][user_id]["Generated Document"]["urls"].append(pdf_url)
        DM.save()

        return {"success": True, "result": {"pdf_url": pdf_url, "filename": pdf_file.filename}}

    except Exception as e:
        Logger.log(f"[EXISTING DOCUMENT LLM] - ERROR: Error saving generated PDF: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"ERROR: Failed to save PDF. {str(e)}"})

# Generate design document endpoint 
@router.post("/generateDesignDocument/{user_id}")
async def generate_design_document(user_id: str):
    """
    Generates a renovation design PDF for existing home owners.
    - Style Analysis image  → room_photo_path  (replaces raw floor plan)
    - Final Image Selection → generated_design_path  (replaces segmented floor plan)
    - Saved Recommendations → furniture_items in room recommendations
    - Preferences sourced from New Home Owner section only
    - Agent -> Outputs -> Generated Floor Plans → shown in PDF if present
    - Agent -> Outputs -> Generated Images      → shown in PDF if present
    """
    tmp_room_photo_path = None
    tmp_generated_design_path = None
    tmp_agent_floor_plan_paths = []
    tmp_agent_generated_image_paths = []

    try:
        if not user_id:
            return JSONResponse(status_code=400, content={"error": "UERROR: One or more required fields are invalid / missing."})

        user = DM.peek(["Users", user_id])
        if user is None:
            return JSONResponse(status_code=404, content={"error": "UERROR: Please login again."})

        # Existing Home Owner data 
        existing_data = DM.peek(["Users", user_id, "Existing Home Owner"])
        if not existing_data:
            return JSONResponse(status_code=404, content={"error": "ERROR: Existing home owner data not found."})

        style_analysis            = existing_data.get("Style Analysis", {})
        room_photo_url            = style_analysis.get("image_url")
        detected_style_from_analysis = style_analysis.get("detected_style", "")

        final_image_selection     = existing_data.get("Final Image Selection", {})
        generated_design_url      = final_image_selection.get("image_url")

        saved_recommendations_data = existing_data.get("Saved Recommendations", {}).get("recommendations", {})
        saved_recommendations = list(saved_recommendations_data.values()) if saved_recommendations_data else []

        # Agent Outputs
        agent_data    = DM.peek(["Users", user_id, "Agent"]) or {}
        agent_outputs = agent_data.get("Outputs", {})

        agent_floor_plan_urls      = agent_outputs.get("Generated Floor Plans", [])
        agent_generated_image_urls = agent_outputs.get("Generated Images", [])

        # Normalise Agent output
        if isinstance(agent_floor_plan_urls, dict):
            agent_floor_plan_urls = list(agent_floor_plan_urls.values())
        if isinstance(agent_generated_image_urls, dict):
            agent_generated_image_urls = list(agent_generated_image_urls.values())

        # Preferences from New Home Owner only
        new_home_owner_data = DM.peek(["Users", user_id, "New Home Owner"])
        budget_min     = "Not specified"
        budget_max     = "Not specified"
        selected_styles     = []
        unit_type       = "Not specified"
        property_type   = "Residential Unit"

        if new_home_owner_data:
            prefs = new_home_owner_data.get("Preferences", {})
            budget_min = prefs.get("budget_min", budget_min)
            budget_max = prefs.get("budget_max", budget_max)
            selected_styles = prefs.get("selected_styles", selected_styles)
            unit_type = prefs.get("unit_type", unit_type)
            property_type = prefs.get("property_type", property_type)
            budget = f"S${budget_min}-S${budget_max}" if budget_min != "Not specified" and budget_max != "Not specified" else "Not specified"

        # Download images
        if room_photo_url:
            try:
                r = requests.get(room_photo_url, timeout=30)
                r.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                    f.write(r.content)
                    tmp_room_photo_path = f.name
            except Exception as e:
                Logger.log(f"[EXISTING DOCUMENT LLM] - ERROR: Error downloading room photo: {e}")

        if generated_design_url:
            try:
                r = requests.get(generated_design_url, timeout=30)
                r.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
                    f.write(r.content)
                    tmp_generated_design_path = f.name
            except Exception as e:
                Logger.log(f"[EXISTING DOCUMENT LLM] - ERROR: Error downloading generated design: {e}")

        # Download Agent-generated floor plans
        if agent_floor_plan_urls:
            tmp_agent_floor_plan_paths = _download_image_list(agent_floor_plan_urls, suffix='.png')
            Logger.log(f"[EXISTING DOCUMENT LLM] - Downloaded {len(tmp_agent_floor_plan_paths)} agent floor plan(s).")

        # Download Agent-generated images
        if agent_generated_image_urls:
            tmp_agent_generated_image_paths = _download_image_list(agent_generated_image_urls, suffix='.jpg')
            Logger.log(f"[EXISTING DOCUMENT LLM] - Downloaded {len(tmp_agent_generated_image_paths)} agent generated image(s).")

        # Build prompt context
        property_type = property_type or "Residential Unit"
        styles_str    = ", ".join(selected_styles) if selected_styles else "Not specified"

        rag_history = RAG.get_history(user_id)
        formatted_chat_history = ""
        if rag_history:
            formatted_chat_history = "CONVERSATION HISTORY:\n"
            for msg in rag_history:
                formatted_chat_history += f"{msg.get('role','unknown').upper()}: {msg.get('content','')}\n"
            formatted_chat_history += "\n"

        saved_rec_summary = ""
        if saved_recommendations:
            saved_rec_summary = "SAVED FURNITURE RECOMMENDATIONS (use these as furniture_items in room recommendations):\n"
            for rec in saved_recommendations:
                name        = rec.get("name", "Item")
                description = rec.get("description", "")
                match       = rec.get("match", "")
                saved_rec_summary += f"- {name} (Match: {match}%): {description}\n"

        system_prompt = (
            "You are an expert interior designer specialising in home renovation in Singapore (2026 market). "
            "You help existing home owners transform their current space into their desired style. "
            "CRITICAL ENCODING RULES - VIOLATIONS WILL CAUSE SYSTEM FAILURE:\n"
            "- Use ONLY these characters: A-Z, a-z, 0-9, basic punctuation (. , ! ? ; : -)\n"
            "- For ALL dashes, use ONLY the ASCII hyphen-minus character (- key on keyboard, character code 45)\n"
            "- NEVER use: en-dash (–), em-dash (—), non-breaking hyphen, or any Unicode dash variants\n"
            "- For ALL quotes, use ONLY straight quotes: apostrophe (') and double quote (\")\n"
            "- NEVER use curly/smart quotes: ' ' " "\n"
            "- Every JSON string value MUST be enclosed in double quotes.\n"
            "- Use the provided saved furniture recommendations as the furniture_items for room recommendations.\n"
            "- ALWAYS suggest specific colors and materials. Never leave them as 'Not specified'.\n"
            "- ENSURE quotation breakdown subtotals add up EXACTLY to the total_quotation.\n"
            "- Conversation history overrides initial form preferences when they conflict.\n"
            "- Focus on renovation and refreshing an existing space, not starting from scratch.\n"
            "- If you generate ANY non-ASCII character, the system will crash."
        )

        user_prompt = f"""Create a comprehensive renovation design document in JSON format for an EXISTING home owner.

PROPERTY DETAILS:
- Unit: {property_type}
- Unit Type: {unit_type}
- Existing Detected Style: {detected_style_from_analysis or 'Not detected'}

CLIENT PREFERENCES (from onboarding):
- Desired Style: {styles_str}
- Colors: Not specified
- Materials: Not specified
- Special Requirements: Not specified

BUDGET: {budget or 'Not specified'}

{saved_rec_summary}

SINGAPORE RENOVATION COST REFERENCE (2026):
HDB BTO/New Units: 3-Room S$36,000-S$45,000 | 4-Room S$51,000-S$65,000 | 5-Room S$67,000-S$83,000
HDB Resale Units: 3-Room S$51,000-S$62,000 | 4-Room S$64,000-S$82,000 | 5-Room S$84,000-S$97,000
Condominiums: New S$40,000-S$75,000 | Resale S$80,000-S$105,000
Landed: New S$100,000-S$250,000+ | Resale S$300,000-S$1.5M+
Component Costs: Carpentry S$250-S$400/ft | Flooring vinyl S$4-S$8 psf / premium S$10-S$47.50 psf
Electrical S$1,500-S$15,000 | Plumbing S$1,500-S$4,000

{formatted_chat_history}

INSTRUCTIONS:
1. The client already lives in this home. Focus on renovation, refreshing, and upgrading.
2. Design to transition from existing style ({detected_style_from_analysis}) to desired style ({styles_str}).
3. For furniture_items in room recommendations, USE the saved furniture recommendations above.
4. ALWAYS provide 3-5 specific color and material suggestions.
5. Quotation breakdown subtotals MUST add up EXACTLY to total_quotation.
6. Use ONLY plain ASCII characters - no special dashes, no curly quotes.

Return ONLY valid JSON matching this exact structure:
{{
  "detected_preferences_override": {{
    "style": "style name",
    "colors": ["Color 1", "Color 2", "Color 3"],
    "materials": ["Material 1", "Material 2", "Material 3"],
    "budget": "budget value",
    "special_notes": "notes or Not specified"
  }},
  "quotation_range": {{
    "minimum_quote": "S$XX,000",
    "maximum_quote": "S$XX,000",
    "recommended_quote": "S$XX,000",
    "quote_basis": "explanation",
    "scope_level": "Light or Moderate or Extensive",
    "cost_factors": ["Factor 1", "Factor 2", "Factor 3"]
  }},
  "quotation_breakdown": {{
    "carpentry":           {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "flooring":            {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "painting":            {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "electrical":          {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "plumbing":            {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "masonry":             {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "furniture_fixtures":  {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "design_consultation": {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "contingency":         {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "total_quotation": "S$XX,000"
  }},
  "executive_summary": {{
    "project_overview": "4-5 sentence overview mentioning existing style, desired style transformation, and renovation goals.",
    "design_philosophy": "3-4 sentences on approach to transition from existing to desired style.",
    "key_recommendations": ["rec1", "rec2", "rec3", "rec4", "rec5"]
  }},
  "design_concept": {{
    "style_direction": "2-3 sentences on transitioning from existing to desired style",
    "color_palette": ["Primary: color (usage)", "Secondary: color (usage)", "Accent: color (usage)"],
    "materials": ["material (where to use)", "material (where to use)", "material (where to use)"],
    "lighting_strategy": "2-3 sentences"
  }},
  "room_recommendations": [
    {{
      "room_name": "Room Name",
      "design_approach": "3-4 sentences on renovation strategy",
      "furniture_items": [
        {{"item": "Name from saved recommendations", "estimated_cost": "S$XXX", "notes": "why it fits"}},
        {{"item": "Name from saved recommendations", "estimated_cost": "S$XXX", "notes": "why it fits"}}
      ],
      "color_specs": "2 sentences",
      "lighting": "2 sentences"
    }}
  ],
  "budget_breakdown": {{
    "total_estimated": "{budget or 'Based on scope'}",
    "buffer_amount": "S$X,000 buffer",
    "by_room": [{{"room": "Room", "amount": "S$X,000", "breakdown": "brief"}}],
    "priority_items": ["Item (S$cost)", "Item (S$cost)"],
    "optional_items": ["Item (S$cost)", "Item (S$cost)"],
    "cost_saving_tips": ["Tip 1", "Tip 2", "Tip 3"]
  }},
  "timeline": {{
    "total_duration": "X weeks",
    "phases": [
      {{
        "phase": "Phase Name",
        "duration": "X weeks",
        "tasks": ["Task 1", "Task 2", "Task 3"],
        "budget_allocation": "S$X,000"
      }}
    ]
  }},
  "next_steps": ["Step 1", "Step 2", "Step 3"],
  "maintenance_guide": {{
    "daily": ["Tip 1", "Tip 2"],
    "monthly": ["Task 1", "Task 2"],
    "annual": ["Task 1", "Task 2"]
  }}
}}"""

        # Call LLM 
        design_data = _call_groq(system_prompt, user_prompt)

        # Build final preferences 
        final_preferences = {"style": styles_str}
        final_preferences, final_budget = apply_conversation_overrides(
            design_data=design_data,
            base_preferences=final_preferences,
            base_budget=budget,
        )

        existing_prefs = existing_data.get("Preferences", {})
        unit_info = {
            "property_type":  existing_prefs.get("property_type", ""),
            "unit_type":      existing_prefs.get("unit_type", ""),
            "selected_styles": existing_prefs.get("selected_styles", []),
            "budget_min":     existing_prefs.get("budget_min"),
            "budget_max":     existing_prefs.get("budget_max"),
        }

        # Generate PDF 
        pdf_buffer = generate_pdf(
            design_data=design_data,
            room_photo_path=tmp_room_photo_path,
            generated_design_path=tmp_generated_design_path,
            preferences=final_preferences,
            budget=final_budget,
            unit_info=unit_info,
            saved_recommendations=saved_recommendations,
            detected_style=detected_style_from_analysis,
            agent_floor_plan_paths=tmp_agent_floor_plan_paths,
            agent_generated_image_paths=tmp_agent_generated_image_paths,
        )

        # Cleanup & return 
        all_tmp_paths = (
            [tmp_room_photo_path, tmp_generated_design_path]
            + tmp_agent_floor_plan_paths
            + tmp_agent_generated_image_paths
        )
        for path in all_tmp_paths:
            if path and os.path.exists(path):
                os.unlink(path)

        pdf_buffer.seek(0)
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=existing_home_design_document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"},
        )

    except Exception as e:
        all_tmp_paths = (
            [tmp_room_photo_path, tmp_generated_design_path]
            + tmp_agent_floor_plan_paths
            + tmp_agent_generated_image_paths
        )
        for path in all_tmp_paths:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass

        Logger.log(f"[EXISTING DOCUMENT LLM] - ERROR: Error generating design document: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"ERROR: Failed to generate design document. {str(e)}"})