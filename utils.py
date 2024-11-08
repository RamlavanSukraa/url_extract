from fastapi import HTTPException
from PIL import Image, ImageOps
import base64
import io
import os
import requests
from datetime import datetime

valid_extensions = ['png', 'jpg', 'jpeg', 'gif', 'webp']

def load_image_from_source(source_path: str):
    """
    Load an image from a URL or local file path.
    """
    try:
        if source_path.startswith("http://") or source_path.startswith("https://"):
            response = requests.get(source_path)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
        else:
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"File not found: {source_path}")
            img = Image.open(source_path)

        return img
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading image: {e}")

def validate_image(image):
    try:
        img = Image.open(image)
        img.verify()  # Verify that it is an image
        if img.format.lower() not in valid_extensions:
            raise ValueError(f"Unsupported image format: {img.format}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    finally:
        if hasattr(image, 'seek'):
            image.seek(0)  # Reset file pointer after verification

def encode_image(image_bytes: bytes):
    try:
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        return encoded_image
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding image: {e}")

def compress_image(image, max_size_mb: float):
    try:
        img = Image.open(image)
        img_format = img.format  # Get the original format

        quality = 95
        buffer = io.BytesIO()
        while True:
            buffer.seek(0)
            img.save(buffer, format=img_format, quality=quality)
            size_kb = buffer.tell() / 1024
            if size_kb <= max_size_mb * 1024 or quality <= 5:
                break  # Stop if the image is within size or quality is too low

            quality -= 5

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        compressed_image_name = f"compressed_image_{timestamp}.{img_format.lower()}"
        compressed_image_path = os.path.join('output', compressed_image_name)

        with open(compressed_image_path, 'wb') as f:
            f.write(buffer.getvalue())

        return compressed_image_path

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error compressing image: {e}")
    finally:
        if hasattr(image, 'seek'):
            image.seek(0)  # Reset file pointer after compression
