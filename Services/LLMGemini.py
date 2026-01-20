"""
LLM Gemini Service

This file handles all interactions with Gemini LLM.
Its responsibility is to generate Stable Diffusion prompts and ensure 
the output is a valid, structured JSON, ready for use by image generation.

Any cleaning, validation, or error handling related to LLM responses is done 
here to ensure that image generation receives clean prompts.
"""

import os
import json
import google.generativeai as genai
# ================================
# JSON extraction and validation
# ================================
def extract_json(text: str) -> dict:
    """
    Extract and validate a JSON object from LLM output.

    LLM responses may contain extra text, markdown formatting, or partially 
    malformed JSON. This function aims to extract the JSON block and validate
    fields.

    Args:
        text: Raw text output from the LLM

    Returns:
        Parsed JSON dictionary containing:
        - prompt
        - negative

    Raises:
        ValueError if a valid JSON object cannot be extracted.
    """
    try:
        text = text.strip()

        # Remove markdown code fences if present (```json ... ```)
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.lower().startswith("json"):
                    text = text[4:]

        text = text.strip()

        # Extract JSON substring
        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")

        parsed = json.loads(text[start:end])

        # Validate required fields
        if "prompt" not in parsed:
            raise ValueError("Missing required field: prompt")


        return parsed

    except json.JSONDecodeError as e:
        # Attempt to recover partially valid JSON
        try:
            if not text.endswith("}"):
                last_quote = text.rfind('"')
                if last_quote > 0:
                    text = text[:last_quote + 1] + "}"

            parsed = json.loads(text[text.find("{"): text.rfind("}") + 1])

            if "prompt" in parsed and "negative" in parsed:
                return parsed
        except:
            pass

        raise ValueError(f"Failed to parse LLM JSON. Raw output: {text[:200]}...") from e

    except Exception as e:
        raise ValueError(f"Unexpected error parsing LLM response: {str(e)}") from e

# ================================
# Gemini model setup
# ================================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

"""
System instruction for Gemini to generate Stable Diffusion prompts is created
in a way that focuses on realistic interior design, avoiding common issues such as
overly cinematic descriptions or camera terminology.

This ensures that the generated prompts are better suited for Stable Diffusion
to create realistic and livable interior design images.

"""
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",
system_instruction=(
    "You are an interior design assistant."

    "Your task is to generate Stable Diffusion prompts for "
    "REALISTIC, LIVABLE INTERIOR DESIGN."

    "RULES:"
    "- Use short, comma-separated phrases (no full sentences)\n"
    "- Keep prompt under 60 tokens\n"
    "- Avoid camera, cinematic, mood, or artistic language\n"
    "- Describe real homes that look professionally designed\n"

    "COLOR & MATERIAL RULES (MANDATORY):"
    "- Always specify wall color and finish\n"
    "- Always specify floor material and tone\n"
    "- Always specify main furniture upholstery color\n"
    "- Always specify rug or textile accent colors\n"
    "- Color palette MUST change according to selected styles\n"

    "PROMPT ORDER (IMPORTANT):"
    "- Walls and finishes first\n"
    "- Flooring second\n"
    "- Furniture colors and materials\n"
    "- Rugs and accent textiles\n"
    "- Room type and style last\n"

    "Return ONLY valid JSON:\n"
    "{ \"prompt\": \"...\" }\n"

    "Do not include markdown, explanations, or extra text."
    )
)

# ================================
# Prompt generation
# ================================
def generate_sd_prompt(styles: str) -> dict:
    """
    Generate a Stable Diffusion prompt using Gemini.

    Workflow:
    1. Sends user-selected styles to the LLM
    2. Enforces JSON-only output
    3. Parses and validates the response
    4. Returns a clean dictionary safe ready for use

    Args:
        styles: User selected style(s)
        
    Returns:
        Dictionary containing:
        - prompt
    """
    prompt = f"""
        Styles: {styles}

        Generate a Stable Diffusion prompt under 60 tokens.

        Requirements:
        - Use comma-separated phrases only
        - Explicitly assign colors to walls, floors, furniture, and rugs
        - Color palette must clearly reflect the selected styles
        - No negative prompt

        Return ONLY valid JSON:
        {{ "prompt": "..." }}
    """


    response = model.generate_content(
        prompt,
        generation_config={
            # temperature is set low to reduce randomness and ensure
            # that the LLM sticks to the system prompt of generating 
            # realistic interior design prompts
            "temperature": 0.3,
            "max_output_tokens": 1024, # fixed token size to prevent overly long outputs
            "response_mime_type": "application/json"
        }
    )

    raw = response.text
    print("GEMINI RAW RESPONSE:\n", raw)

    # Ensure image generation always receives valid JSON
    return extract_json(raw)
