import os
import json
import google.generativeai as genai

def extract_json(text: str) -> dict:
    try:
        text = text.strip()

        # Remove markdown code fences
        if text.startswith("```"):
            text = text.split("```")[1].strip()

        # Remove leading 'json' token (Gemini quirk)
        if text.lower().startswith("json"):
            text = text[4:].strip()

        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end == -1:
            raise ValueError("No JSON object found")

        return json.loads(text[start:end])

    except Exception as e:
        raise ValueError(f"Failed to parse LLM JSON: {text}") from e


# ─── Gemini setup ─────────────────────────────────────────────

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=(
        "You generate Stable Diffusion prompts. "
        "Use short, comma-separated visual tags. "
        "Focus on materials, lighting, camera, and style. "
        "Return valid JSON only with no extra text. "
        "Keep prompts concise (under 100 words total). "
        "If unsafe, return {\"prompt\":\"\",\"negative\":\"\"}."
    )
)

def generate_sd_prompt(styles: str, preferences: str) -> str:
    prompt = f"""
Styles: {styles}
Preferences: {preferences}

Generate a Stable Diffusion prompt and negative prompt.

Return ONLY valid JSON (no markdown, no extra text):
{{ "prompt": "...", "negative": "..." }}

Keep both prompts concise.
"""

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.4,
            "max_output_tokens": 2048,  # Increased from 512
            "response_mime_type": "application/json"
        }
    )

    raw = response.text
    print("GEMINI RAW RESPONSE:\n", raw)

    return raw