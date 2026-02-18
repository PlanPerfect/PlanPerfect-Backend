import os
from google import genai
from google.genai import types
from PIL import Image
import io
import traceback
from Services import Logger

# Gemini model setup
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

image_gen_model = "gemini-3-pro-image-preview"


def generate_interior_design(
    init_image: Image.Image,
    styles: str,
    user_modifications: str = None,
    furniture_images: list = None  # list of PIL Images to use as references
) -> Image.Image:
    """
    Generate interior design transformation using Gemini Imagen 3.

    Imagen 3 is powerful enough to handle both initial generation and regeneration
    directly without needing any vision model analysis.

    Args:
        init_image:         Original room image
        styles:             Base style string (e.g., "Modern, Minimalist")
        user_modifications: Optional user-specific changes (e.g., "Make walls darker, add plants")
        furniture_images:   Optional list of PIL Images to use as reference items
                            (e.g., selected product recommendations from the user)

    Returns:
        PIL Image object of the generated design
    """
    try:
        # Calculate aspect ratio from input image - use original dimensions
        width, height = init_image.size
        aspect_ratio = width / height

        # Use the original image aspect ratio by selecting closest match
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

        # Validate and truncate user modifications if too long
        if user_modifications and len(user_modifications) > 200:
            user_modifications = user_modifications[:200]

        has_furniture = furniture_images and len(furniture_images) > 0

        if user_modifications and user_modifications.strip():
            transformation_prompt = f"""
            The FIRST image is the room to redesign. Redesign it in {styles} style.
            {user_modifications.strip()}

            Keep the same room layout and furniture positions. Create a photorealistic interior design.
            """
        else:
            transformation_prompt = f"""
            The FIRST image is the room to redesign. Transform it to {styles} style.

            REQUIREMENTS:
            1. Preserve the exact room layout and furniture positions
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

        # Append furniture reference instructions to prompt if applicable
        if has_furniture:
            transformation_prompt += f"""
            REFERENCE ITEMS: The {len(furniture_images)} image(s) that follow after the room image are PRODUCT REFERENCE PHOTOS only.
            - DO NOT redesign or output these reference images
            - DO NOT treat them as rooms or spaces
            - USE them only to understand the style, color, shape, and form of specific furniture or decor items
            - Incorporate items visually similar to these references naturally into the redesigned room
            """

        # prompt first, then room image, then furniture references
        contents = [transformation_prompt, init_image]

        if has_furniture:
            for img in furniture_images:
                contents.append(img)

        Logger.log(
            f"[LLM GEMINI] - Generating design | styles={styles} | "
            f"furniture_refs={len(furniture_images) if has_furniture else 0} | "
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

        # Extract the generated image
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