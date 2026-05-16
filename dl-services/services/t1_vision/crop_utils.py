# dl-services/services/t1_vision/crop_utils.py

from PIL import Image

def crop_image(image: Image.Image, box: dict) -> Image.Image:
    """
    Cắt ảnh theo bounding box.
    box: {'xmin': float, 'ymin': float, 'xmax': float, 'ymax': float}
    Trả về ảnh đã crop.
    """
    return image.crop((box['xmin'], box['ymin'], box['xmax'], box['ymax']))