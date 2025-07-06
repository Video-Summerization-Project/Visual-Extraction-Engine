import base64
from io import BytesIO
from PIL import Image

def image_to_base64(image_path):
    """Convert an image to a base64-encoded string along with its MIME type."""
    try:
        img = Image.open(image_path)
        buffered = BytesIO()
        img_format = img.format if img.format else "JPEG"
        img.save(buffered, format=img_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        mime_type = f"image/{img_format.lower()}"
        return img_str, mime_type
    except Exception as e:
        print(f"Error converting image to base64: {str(e)}")
        return None, None
