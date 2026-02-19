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
from Services.NewHomeOwnerDesignDocsPdf import generate_pdf
from Services import Logger

router = APIRouter(prefix="/designDocument/newHomeOwnerDocumentLlm", tags=["LLM PDF Generation - New Home Owner"], dependencies=[Depends(_verify_api_key)])

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── Unicode cleaning ───────────────────────────────────────────────────────────
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


# ── JSON extraction & repair ───────────────────────────────────────────────────
def _extract_json_str(raw: str) -> str:
    """Pull JSON out of markdown code fences if present."""
    raw = raw.strip()
    if "```json" in raw:
        return raw.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in raw:
        return raw.split("```", 1)[1].split("```", 1)[0].strip()
    return raw


def _attempt_repair(bad_json: str) -> dict:
    """
    Try a series of increasingly aggressive repairs on malformed JSON.
    Returns parsed dict or raises ValueError if all attempts fail.
    """
    attempts = [bad_json]

    # 1. Fix unquoted string values like:  "phase": Handover & Styling"
    #    Pattern: colon followed by whitespace then an unquoted word/phrase until a comma/brace/bracket
    repaired = re.sub(
        r':\s*([A-Za-z][^",\}\]\n]{0,120})"',
        lambda m: ': "' + m.group(1).replace('"', "'") + '"',
        bad_json,
    )
    attempts.append(repaired)

    # 2. Remove trailing commas before } or ]
    no_trailing = re.sub(r',\s*([}\]])', r'\1', bad_json)
    attempts.append(no_trailing)

    # 3. Combine both repairs
    both = re.sub(r',\s*([}\]])', r'\1', repaired)
    attempts.append(both)

    for attempt in attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue

    raise ValueError("All JSON repair attempts failed.")


def _parse_llm_response(response_text: str) -> dict:
    """
    Clean → extract → parse JSON from an LLM response string.
    Falls back to repair logic if initial parse fails.
    """
    cleaned = _clean_text(response_text)
    json_str = _extract_json_str(cleaned)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        Logger.log("[NEW DOCUMENT LLM] - ERROR: Initial JSON parse failed, attempting repair...")
        return _attempt_repair(json_str)


# ── Groq call with failed_generation fallback ─────────────────────────────────
def _call_groq(system_prompt: str, user_prompt: str) -> dict:
    """
    Call Groq with json_object mode.
    If Groq returns json_validate_failed (400), extract the failed_generation
    string from the error body and attempt to parse/repair it ourselves.
    """
    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            model="openai/gpt-oss-120b",
            temperature=0.7,
            max_tokens=5120,
            response_format={"type": "json_object"},
        )
        return _parse_llm_response(completion.choices[0].message.content)

    except Exception as groq_err:
        # Try to salvage the failed_generation from the Groq error body
        err_str = str(groq_err)
        failed_gen = None

        # Groq SDK surfaces the raw response body in the exception message
        try:
            # The error message contains the JSON body as a string; parse it
            match = re.search(r"'failed_generation':\s*'(.*?)'(?:\s*\})", err_str, re.DOTALL)
            if match:
                failed_gen = match.group(1).encode("utf-8").decode("unicode_escape")
            else:
                # Try parsing the whole error as JSON
                body_match = re.search(r"\{.*\}", err_str, re.DOTALL)
                if body_match:
                    body = json.loads(body_match.group(0).replace("'", '"'))
                    failed_gen = body.get("error", {}).get("failed_generation")
        except Exception:
            pass

        if failed_gen:
            Logger.log("[NEW DOCUMENT LLM] - ERROR: Groq json_validate_failed — attempting to parse failed_generation...")
            try:
                return _parse_llm_response(failed_gen)
            except Exception as repair_err:
                Logger.log(f"[NEW DOCUMENT LLM] - ERROR: Repair also failed: {repair_err}")

        # Re-raise original so the endpoint's except block can handle it
        raise groq_err


# ── Placeholder helper ─────────────────────────────────────────────────────────
def _is_placeholder_value(value):
    if not value:
        return True
    placeholder_phrases = {
        'not specified', 'none',
        'open to designer recommendations',
        'open to recommendations', '',
    }
    return str(value).lower().strip() in placeholder_phrases


# ── Conversation override helper ───────────────────────────────────────────────
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


# ── Save PDF endpoint ──────────────────────────────────────────────────────────
@router.post("/savePdf/{user_id}")
async def save_generated_pdf(user_id: str, pdf_file: UploadFile = File(...)):
    try:
        if not user_id:
            return JSONResponse(status_code=400, content={"error": "UERROR: One or more required fields are invalid / missing."})

        user = DM.peek(["Users", user_id])
        if user is None:
            return JSONResponse(status_code=404, content={"error": "UERROR: Please login again."})

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_file.filename = f"design_document_{timestamp}.pdf"

        upload_result = FM.store_file(file=pdf_file, subfolder=f"newHomeOwners/{user_id}/documents")
        pdf_url = upload_result["url"]

        if "Generated Document" not in DM.data["Users"][user_id]:
            DM.data["Users"][user_id]["Generated Document"] = {}

        if not isinstance(DM.data["Users"][user_id]["Generated Document"].get("urls"), list):
            DM.data["Users"][user_id]["Generated Document"]["urls"] = []

        DM.data["Users"][user_id]["Generated Document"]["urls"].append(pdf_url)
        DM.save()

        return {"success": True, "result": {"pdf_url": pdf_url, "filename": pdf_file.filename}}

    except Exception as e:
        Logger.log(f"[NEW DOCUMENT LLM] - ERROR: Error saving generated PDF: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"ERROR: Failed to save PDF. {str(e)}"})


# ── Generate design document endpoint ─────────────────────────────────────────
@router.post("/generateDesignDocument/{user_id}")
async def generate_design_document(user_id: str):
    tmp_floor_plan_path = None
    tmp_segmented_path = None

    try:
        if not user_id:
            return JSONResponse(status_code=400, content={"error": "UERROR: One or more required fields are invalid / missing."})

        user = DM.peek(["Users", user_id])
        if user is None:
            return JSONResponse(status_code=404, content={"error": "UERROR: Please login again."})

        user_data = DM.peek(["Users", user_id, "New Home Owner"])
        if not user_data:
            return JSONResponse(status_code=404, content={"error": "ERROR: User data not found."})

        # ── Extract preferences & unit info ──────────────────────────────────
        preferences_data  = user_data.get("Preferences", {})
        budget            = preferences_data.get("budget", "Not specified")
        styles            = preferences_data.get("Preferred Styles", {}).get("styles", [])

        floor_plan_url         = user_data.get("Uploaded Floor Plan", {}).get("url")
        segmented_floor_plan_url = user_data.get("Segmented Floor Plan", {}).get("url")

        unit_info_data = user_data.get("Unit Information", {})
        unit_rooms     = unit_info_data.get("unit", "Residential Unit")
        unit_type      = unit_info_data.get("unitType", "Not specified")
        unit_size      = unit_info_data.get("unitSize", "Not specified")

        number_of_rooms = unit_info_data.get("Number Of Rooms", {})
        room_counts = {
            "LIVING":   number_of_rooms.get("livingRoom", 0),
            "KITCHEN":  number_of_rooms.get("kitchen",    0),
            "BATH":     number_of_rooms.get("bathroom",   0),
            "BEDROOM":  number_of_rooms.get("bedroom",    0),
            "LEDGE":    number_of_rooms.get("ledge",      0),
            "BALCONY":  number_of_rooms.get("balcony",    0),
        }

        # ── Download images ───────────────────────────────────────────────────
        for url, attr_name, suffix in [
            (floor_plan_url,           'tmp_floor_plan_path', '.png'),
            (segmented_floor_plan_url, 'tmp_segmented_path',  '.png'),
        ]:
            if url:
                try:
                    r = requests.get(url, timeout=30)
                    r.raise_for_status()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                        f.write(r.content)
                        if attr_name == 'tmp_floor_plan_path':
                            tmp_floor_plan_path = f.name
                        else:
                            tmp_segmented_path = f.name
                except Exception as e:
                    Logger.log(f"[NEW DOCUMENT LLM] - ERROR: Error downloading {attr_name}: {e}")

        # ── Build prompt context ──────────────────────────────────────────────
        rooms_summary = [f"{c} {rt.title()}" for rt, c in room_counts.items() if c and c > 0]
        property_type = unit_rooms or "Residential Unit"
        styles_str    = ", ".join(styles) if styles else "Not specified"

        rag_history = RAG.get_history(user_id)
        formatted_chat_history = ""
        if rag_history:
            formatted_chat_history = "CONVERSATION HISTORY:\n"
            for msg in rag_history:
                formatted_chat_history += f"{msg.get('role','unknown').upper()}: {msg.get('content','')}\n"
            formatted_chat_history += "\n"

        room_details = ("Identified Rooms:\n" + ", ".join(rooms_summary)) if rooms_summary else ""

        system_prompt = (
            "You are an expert interior designer with deep knowledge of Singapore's renovation market in 2026. "
            "You create detailed, professional design documents in strict JSON format. "
            "CRITICAL RULES:\n"
            "- Use ONLY plain ASCII hyphens (-) for dashes. NEVER use en-dash (\\u2013), em-dash (\\u2014), "
            "or non-breaking hyphen (\\u2011).\n"
            "- Use ONLY straight apostrophes (') and straight double quotes (\"). NEVER use curly/smart quotes.\n"
            "- Every JSON string value MUST be enclosed in double quotes.\n"
            "- ALWAYS suggest specific colors and materials. Never leave them as 'Not specified'.\n"
            "- ENSURE quotation breakdown subtotals add up EXACTLY to the total_quotation.\n"
            "- Conversation history overrides initial form preferences when they conflict."
        )

        user_prompt = f"""Create a comprehensive interior design document in JSON format.

PROPERTY DETAILS:
- Unit: {property_type}
- Unit Type: {unit_type}
- Unit Size: {unit_size}
- Room Layout: {room_details or 'Standard residential layout'}

CLIENT PREFERENCES (Initial):
- Style: {styles_str}
- Colors: Not specified
- Materials: Not specified
- Special Requirements: Not specified

BUDGET: {budget or 'Not specified'}

SINGAPORE RENOVATION COST REFERENCE (2026):
HDB BTO/New Units: 3-Room S$36,000-S$45,000 | 4-Room S$51,000-S$65,000 | 5-Room S$67,000-S$83,000
HDB Resale Units: 3-Room S$51,000-S$62,000 | 4-Room S$64,000-S$82,000 | 5-Room S$84,000-S$97,000
Condominiums: New S$40,000-S$75,000 | Resale S$80,000-S$105,000
Landed: New S$100,000-S$250,000+ | Resale S$300,000-S$1.5M+
Component Costs: Carpentry S$250-S$400/ft | Flooring vinyl S$4-S$8 psf / premium S$10-S$47.50 psf
Electrical S$1,500-S$15,000 | Plumbing S$1,500-S$4,000

{formatted_chat_history}

INSTRUCTIONS:
1. Analyze conversation history and override initial preferences if user expressed different ones.
2. ALWAYS provide 3-5 specific color and material suggestions. Never say "Not specified".
3. Provide a detailed quotation breakdown where subtotals add up EXACTLY to total_quotation.
4. Use ONLY plain ASCII characters in all string values - no special dashes, no curly quotes.

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
    "carpentry":          {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "flooring":           {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "painting":           {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "electrical":         {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "plumbing":           {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "masonry":            {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "furniture_fixtures": {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "design_consultation":{{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "contingency":        {{"description": "...", "quantity": "...", "unit_cost": "...", "subtotal": "S$X,000"}},
    "total_quotation": "S$XX,000"
  }},
  "executive_summary": {{
    "project_overview": "4-5 sentence overview",
    "design_philosophy": "3-4 sentences",
    "key_recommendations": ["rec1", "rec2", "rec3", "rec4", "rec5"]
  }},
  "space_analysis": {{
    "total_area": "{unit_size}",
    "room_breakdown": [
      {{"room_name": "Living Room", "analysis": "2-3 sentences"}},
      {{"room_name": "Kitchen", "analysis": "2-3 sentences"}},
      {{"room_name": "Bedroom", "analysis": "2-3 sentences"}},
      {{"room_name": "Bathroom", "analysis": "2-3 sentences"}}
    ]
  }},
  "design_concept": {{
    "style_direction": "2-3 sentences",
    "color_palette": ["Primary: color (usage)", "Secondary: color (usage)", "Accent: color (usage)"],
    "materials": ["material (where to use)", "material (where to use)", "material (where to use)"],
    "lighting_strategy": "2-3 sentences"
  }},
  "room_recommendations": [
    {{
      "room_name": "Room Name",
      "design_approach": "3-4 sentences",
      "furniture_items": [
        {{"item": "Item name", "estimated_cost": "S$XXX", "notes": "justification"}},
        {{"item": "Item name", "estimated_cost": "S$XXX", "notes": "justification"}}
      ],
      "color_specs": "2 sentences",
      "lighting": "2 sentences"
    }}
  ],
  "budget_breakdown": {{
    "total_estimated": "{budget or 'Based on scope'}",
    "buffer_amount": "S$X,000 buffer",
    "by_room": [{{"room": "Room", "amount": "S$X,000", "breakdown": "brief"}}],
    "priority_items": ["Item ($cost)", "Item ($cost)"],
    "optional_items": ["Item ($cost)", "Item ($cost)"],
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

        # ── Call LLM ─────────────────────────────────────────────────────────
        design_data = _call_groq(system_prompt, user_prompt)

        # ── Build final preferences & unit info ───────────────────────────────
        final_preferences = {"style": styles_str, "styles": styles}
        final_preferences, final_budget = apply_conversation_overrides(
            design_data=design_data,
            base_preferences=final_preferences,
            base_budget=budget,
        )

        unit_info = {
            "unit_rooms":  unit_rooms,
            "unit_types":  [unit_type] if unit_type != "Not specified" else [],
            "unit_sizes":  [unit_size] if unit_size != "Not specified" else [],
            "room_counts": room_counts,
        }

        # ── Generate PDF ──────────────────────────────────────────────────────
        pdf_buffer = generate_pdf(
            design_data=design_data,
            raw_floor_plan_path=tmp_floor_plan_path,
            segmented_floor_plan_path=tmp_segmented_path,
            preferences=final_preferences,
            budget=final_budget,
            unit_info=unit_info,
        )

        # ── Cleanup & return ──────────────────────────────────────────────────
        for path in [tmp_floor_plan_path, tmp_segmented_path]:
            if path and os.path.exists(path):
                os.unlink(path)

        pdf_buffer.seek(0)
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=interior_design_documentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"},
        )

    except Exception as e:
        for path in [tmp_floor_plan_path, tmp_segmented_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass

        Logger.log(f"[NEW DOCUMENT LLM] - ERROR: Error generating design document: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"ERROR: Failed to generate design document. {str(e)}"})