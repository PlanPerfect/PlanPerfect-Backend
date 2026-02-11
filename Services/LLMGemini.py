import os
from google import genai
from google.genai import types
from PIL import Image
import io

# ================================
# Gemini model setup
# ================================
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Vision model for analysis
vision_model_name = "gemini-2.5-pro"

# - "gemini-2.5-flash-image" (faster, cheaper)
# - "gemini-3-pro-image-preview" (better quality, advanced reasoning, text rendering)
image_gen_model = "gemini-3-pro-image-preview"

# Image generation using Nano Banana (Gemini native)
def generate_interior_design(init_image: Image.Image, styles: str) -> Image.Image:
    # Calculate aspect ratio from input image - use original dimensions
    width, height = init_image.size
    aspect_ratio = width / height
    
    # Use the original image aspect ratio by selecting closest match
    # But prefer to keep it as close to original as possible
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
    
    print(f"Original image size: {init_image.size}, Aspect ratio: {aspect_ratio:.2f}, Using: {aspect_ratio_str}")
    
    # Step 1: Analyze the room with Gemini Vision
    analysis_prompt = f"""
    Analyze this room image and describe:
    1. Current style and color scheme
    2. EXACT furniture pieces and their positions (left/right/center, specific placement)
    3. Flooring type and walls
    4. Room dimensions and architectural features (windows, doors, ceiling)
    
    Keep it concise but SPECIFIC about furniture placement, 3-4 sentences maximum.
    """
    
    analysis_response = client.models.generate_content(
        model=vision_model_name,
        contents=[analysis_prompt, init_image],
        config=types.GenerateContentConfig(
            temperature=0.3
        )
    )
    
    room_analysis = analysis_response.text
    print(f"Room Analysis: {room_analysis}")
    
    # Step 2: Create transformation prompt for Nano Banana
    transformation_prompt = f"""
    Transform this interior room to {styles} style.
    
    Current room: {room_analysis}
    
    CRITICAL REQUIREMENTS - DO NOT DEVIATE:
    1. PRESERVE THE EXACT LAYOUT: Keep all furniture in their current positions
    2. MAINTAIN ROOM STRUCTURE: Same room dimensions, window placement, door locations
    3. KEEP FURNITURE COUNT: Same number of furniture pieces in same locations
    4. PRESERVE SPATIAL RELATIONSHIPS: If sofa is facing coffee table, keep that relationship
    
    WHAT YOU CAN CHANGE:
    - Wall colors and textures (paint, wallpaper) to match {styles}
    - Flooring materials and colors to suit {styles} aesthetic
    - Furniture upholstery, cushions, and fabric colors
    - Decorative elements: artwork, plants, accessories
    - Lighting fixtures style (but keep positions)
    - Window treatments (curtains, blinds) in {styles} style
    
    WHAT YOU MUST NOT CHANGE:
    - Room layout and dimensions
    - Furniture positions and placement
    - Number of furniture pieces
    - Basic furniture shapes (a sofa stays a sofa, table stays a table)
    - Architectural features (windows, doors, built-ins)
    
    Create a photorealistic interior design transformation that looks like the SAME ROOM but redesigned in {styles} style.
    Think: "before and after renovation of the same space" not "completely different room".
    """
    
    print(f"Transformation Prompt: {transformation_prompt}")
    
    # Step 3: Generate image with Nano Banana
    response = client.models.generate_content(
        model=image_gen_model,
        contents=[transformation_prompt, init_image],
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
            print(f"Generated image size: {generated_image.size}")
            return generated_image
    
    raise Exception("No image generated in response")


# ================================
# Alternative: Using chat mode for iterative editing
# ================================
def generate_interior_design_with_mask(
    init_image: Image.Image, 
    styles: str,
    preserve_layout: bool = True
) -> Image.Image:
    # Calculate aspect ratio from input image
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
    
    print(f"Original image size: {init_image.size}, Aspect ratio: {aspect_ratio:.2f}, Using: {aspect_ratio_str}")
    
    # Analyze the room first
    analysis_prompt = """
    Briefly describe this room's current style, layout, and key furniture with their exact positions.
    Be specific about where each piece of furniture is located.
    Keep it to 3 sentences.
    """
    
    analysis_response = client.models.generate_content(
        model=vision_model_name,
        contents=[analysis_prompt, init_image],
        config=types.GenerateContentConfig(temperature=0.3)
    )
    
    room_analysis = analysis_response.text
    print(f"Room Analysis: {room_analysis}")
    
    # Create appropriate prompt based on preserve_layout flag
    if preserve_layout:
        edit_prompt = f"""
        Transform this room to {styles} interior design style.
        
        Current room: {room_analysis}
        
        STRICT REQUIREMENTS - ABSOLUTELY PRESERVE:
        1. EXACT same room layout and furniture placement
        2. Same number of furniture pieces in same positions
        3. Same architectural features (windows, doors, ceiling height)
        4. Same spatial relationships between furniture
        
        ONLY CHANGE THESE ELEMENTS:
        - Wall colors and textures (to match {styles})
        - Flooring materials and colors (to match {styles})
        - Furniture upholstery, fabrics, and finishes (to match {styles})
        - Decorative elements: artwork, plants, accessories (to match {styles})
        - Lighting fixture styles (keep same positions)
        - Window treatments (curtains, blinds in {styles} style)
        
        The transformation should look like a professional interior designer redecorated
        the SAME EXACT ROOM in {styles} style. Same layout, different styling.
        """
    else:
        edit_prompt = f"""
        Transform this room to {styles} interior design style.
        
        Current room: {room_analysis}
        
        Maintain the general room dimensions and layout concept.
        Apply {styles} aesthetic comprehensively:
        - Update furniture to match the style
        - Change colors, materials, and finishes
        - Add appropriate decor and accessories
        - Adjust lighting to suit the aesthetic
        
        Create a cohesive, professional {styles} interior design.
        """
    
    print(f"Edit Prompt: {edit_prompt}")
    
    # Generate the transformed image
    response = client.models.generate_content(
        model=image_gen_model,
        contents=[edit_prompt, init_image],
        config=types.GenerateContentConfig(
            response_modalities=['IMAGE'],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio_str,
                image_size="2K"
            )
        )
    )
    
    # Extract generated image
    for part in response.parts:
        if part.inline_data is not None:
            image_data = part.inline_data.data
            generated_image = Image.open(io.BytesIO(image_data))
            print(f"Generated image size: {generated_image.size}")
            return generated_image
    
    raise Exception("No image generated in response")