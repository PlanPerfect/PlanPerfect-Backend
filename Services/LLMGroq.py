import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_sd_prompt(styles: str, preferences: str) -> str:
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are an expert interior designer creating Stable Diffusion prompts."
            },
            {
                "role": "user",
                "content": f"""
Styles: {styles}
Preferences: {preferences}

Generate:
1. A detailed Stable Diffusion interior design prompt
2. A negative prompt
Return JSON:
{{ "prompt": "...", "negative": "..." }}
"""
            }
        ],
        temperature=0.7
    )

    return completion.choices[0].message.content
