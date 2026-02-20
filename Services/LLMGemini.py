import os
from google import genai
from google.genai import types
from PIL import Image
import io
import traceback
from Services import Logger

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

image_gen_model = "gemini-3-pro-image-preview"
image_gen_model_fallback = "gemini-2.5-flash-image"


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

        base_prompt = f""" You are a photo editing tool. Your ONLY job is to RECOLOR and RESTYLE the room in the reference image. You are NOT allowed to redesign or reimagine it.
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
            transformation_prompt = base_prompt

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
                f"""The following {len(furniture_images)} image(s) are FURNITURE REFERENCE images.
                IMPORTANT INSTRUCTIONS FOR THESE REFERENCE IMAGES: 
                Each image may show a furniture item inside a room scene or environment.
                You must EXTRACT and identify ONLY the specific furniture item from each image. 
                IGNORE the room, background, walls, floors, and any surrounding objects in these reference images.
                IGNORE the style or aesthetic of any room shown in the reference images.
                Focus ONLY on the furniture piece itself: its shape, form, and structure.
                Incorporate ONLY that furniture item into the redesigned room, restyled to match the target aesthetic.
                Do NOT let the reference image room influence the output room style in any way.
            """
            )
            for img in furniture_images:
                contents.append(img)

        Logger.log(
            f"[LLM GEMINI] - Generating design | styles={styles} | "
            f"furniture_refs={len(furniture_images) if has_furniture else 0} | "
            f"descriptions={len(furniture_descriptions) if has_descriptions else 0} | "
        )

        if has_descriptions:
            Logger.log(f"[LLM GEMINI] - Furniture descriptions: {furniture_descriptions}")

        def _call_model(model_name):
            Logger.log(f"[LLM GEMINI] - Calling model: {model_name}")
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['IMAGE'],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio_str,
                        image_size="2K"
                    )
                )
            )

        try:
            response = _call_model(image_gen_model)
            Logger.log(f"[LLM GEMINI] - Model {image_gen_model} responded successfully")
        except Exception as primary_err:
            primary_str = str(primary_err)
            if any(k in primary_str for k in ["503", "UNAVAILABLE", "high demand", "overloaded", "quota"]):
                Logger.log(f"[LLM GEMINI] - Primary model ({image_gen_model}) unavailable: {primary_str[:120]}")
                Logger.log(f"[LLM GEMINI] - Switching to fallback model: {image_gen_model_fallback}")
                response = _call_model(image_gen_model_fallback)
                Logger.log(f"[LLM GEMINI] - Fallback model {image_gen_model_fallback} responded successfully")
            else:
                raise primary_err

        for part in response.parts:
            if part.inline_data is not None:
                image_data = part.inline_data.data
                generated_image = Image.open(io.BytesIO(image_data))
                Logger.log(f"[LLM GEMINI] - Image generated successfully")
                return generated_image

        raise Exception("No image generated in response")

    except Exception as e:
        Logger.log(
            f"[LLM GEMINI] - ERROR in generate_interior_design: {str(e)}. "
            f"Error type: {type(e).__name__}. Full traceback:\n{traceback.format_exc()}"
        )
        raise