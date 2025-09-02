from PIL import Image
import io
import numpy as np

def load_image(file_storage) -> Image.Image:
    """Load a PIL Image from a Flask `FileStorage` upload."""
    file_storage.stream.seek(0)
    img = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    return img

def pil_to_tensor(img):
    """Return a PIL Image for CNN processing.
    The CNN model will handle the tensor conversion internally.
    """
    # Return the PIL image directly - the CNN model will apply transforms
    return img

def preprocess_for_cnn(img: Image.Image) -> Image.Image:
    """Additional preprocessing for CNN if needed"""
    # Ensure RGB format
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Optional: resize very large images to reasonable size before CNN processing
    max_size = 1024
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    return img
