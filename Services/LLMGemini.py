import os
from google import genai
from google.genai import types
from PIL import Image
import io
import traceback
from Services import Logger

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

image_gen_model = "gemini-3-pro-image-preview"


def generate_interior_design(
    init_image: Image.Image,
    styles: str,
    user_modifications: str = None,
    furniture_images: list = None,
    furniture_descriptions: list = None
) -> Image.Image:
    try:
        width, height = init_image.size
        aspect_ratio = width / height

        if aspect_ratio >= 1.6:
            aspect_ratio_str = "16:9"
        elif aspect_ratio >= 1.25:
            aspect_ratio_str = "4:3"
        elif aspect_ratio >= 0.8:
            aspect_ratio_str = "1:1"
        elif aspect_ratio >= 0.65:
            aspect_ratio_str = "3:4"
        else:
            aspect_ratio_str = "9:16"

        if user_modifications and len(user_modifications) > 200:
            user_modifications = user_modifications[:200]

        has_furniture = furniture_images and len(furniture_images) > 0
        has_descriptions = furniture_descriptions and len(furniture_descriptions) > 0

        base_prompt = f"""
            The FIRST image is the original room.

            Apply {styles} styling to this EXACT room.

            CRITICAL INSTRUCTION:
            This is a RESTYLING task, NOT a redesign.

            You must preserve the room exactly as it is.

            NON-NEGOTIABLE STRUCTURE RULES:
            - DO NOT change room dimensions
            - DO NOT move walls, windows, or doors
            - DO NOT change ceiling structure
            - DO NOT change camera angle or perspective
            - DO NOT zoom in or out
            - DO NOT crop the image differently

            LAYOUT RULES:
            - Keep ALL furniture in the EXACT same positions
            - Keep the same number of furniture pieces
            - Do NOT remove any furniture
            - Do NOT add new large furniture pieces
            - Do NOT replace furniture with different types
            - Preserve sofa shapes, cabinet layout, island placement, etc.

            WHAT YOU ARE ALLOWED TO CHANGE:
            - Wall colours and finishes
            - Flooring materials (keep layout identical)
            - Furniture upholstery and surface materials
            - Decorative styling (art, cushions, small decor)
            - Lighting fixture style (keep exact positions)
            - Textures and materials to reflect {styles} aesthetic

            The final result must look like:
            "The SAME room, photographed from the SAME angle,
            with the SAME layout and furniture placement,
            but professionally restyled in {styles} aesthetic."

            Photorealistic. Natural lighting. Realistic materials.
        """

        if user_modifications and user_modifications.strip():
            base_prompt += f"""

            USER REQUESTED ADDITIONAL CHANGES:
            {user_modifications.strip()}

            These requests must follow ALL structural and layout preservation rules above.
            """
                    
            transformation_prompt = base_prompt

        else:
            transformation_prompt = f"""
                The FIRST image is the room to redesign. Transform it to {styles} style.

                REQUIREMENTS:
                1. Preserve the EXACT room layout and furniture positions
                2. Maintain the same room structure (dimensions, windows, doors)
                3. Keep the same number of furniture pieces in their current locations
                4. Update only the styling elements:
                - Wall colors and textures for {styles} aesthetic
                - Flooring materials and colors
                - Furniture upholstery and finishes
                - Decorative elements (artwork, plants, accessories)
                - Lighting fixture styles (keep positions)
                - Window treatments in {styles} style

                Generate a photorealistic transformation that looks like the SAME ROOM professionally redesigned in {styles} style.
            """

        if has_descriptions:
            desc_text = "\n".join(f"- {d}" for d in furniture_descriptions if d)
            transformation_prompt += f"""
                MANDATORY FURNITURE REQUIREMENTS — YOU MUST FOLLOW THESE EXACTLY:
                The user has specifically selected the following items. These are NON-NEGOTIABLE.
                Every single item listed below MUST appear visibly and clearly in the final image:
                {desc_text}

                STRICT RULES:
                - Do NOT substitute any of these items with alternatives
                - Do NOT omit any of these items
                - Each item must be clearly recognizable in the final room
                - Place them naturally within the room layout
                - Style them to match the {styles} aesthetic but preserve their core form and identity
            """

        contents = [transformation_prompt, init_image]

        if has_furniture:
            contents.append(
                f"The following {len(furniture_images)} image(s) are supplementary visual references only. "
                "DO NOT redesign these images. They are provided for additional visual context of the selected items above."
            )
            for img in furniture_images:
                contents.append(img)

        Logger.log(
            f"[LLM GEMINI] - Generating design | styles={styles} | "
            f"furniture_refs={len(furniture_images) if has_furniture else 0} | "
            f"descriptions={len(furniture_descriptions) if has_descriptions else 0} | "
            f"aspect_ratio={aspect_ratio_str}"
        )

        response = client.models.generate_content(
            model=image_gen_model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE'],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio_str,
                    image_size="2K"
                )
            )
        )

        for part in response.parts:
            if part.inline_data is not None:
                image_data = part.inline_data.data
                generated_image = Image.open(io.BytesIO(image_data))
                return generated_image

        raise Exception("No image generated in response")

    except Exception as e:
        Logger.log(
            f"[LLM GEMINI] - ERROR in generate_interior_design: {str(e)}. "
            f"Error type: {type(e).__name__}. Full traceback:\n{traceback.format_exc()}"
        )
        raise