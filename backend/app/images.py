import base64
import io
import warnings

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from backend.app.config import Settings


def decode_image(data: bytes, settings: Settings) -> Image.Image:
    """Check image format and size before decoding pixels."""
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "Image exceeds the 8 MiB upload limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise HTTPException(415, "Only JPEG and PNG image content is supported.")
                if max(image.size) > settings.max_image_dimension or (
                    image.width * image.height > settings.max_image_pixels
                ):
                    raise HTTPException(
                        413, "Image dimensions exceed 4096 pixels or 12 megapixels."
                    )
                if getattr(image, "n_frames", 1) != 1:
                    raise HTTPException(415, "Animated images are not supported.")
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                return image.convert("RGB")
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise HTTPException(413, "Decoded image is too large.") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(400, "The file is not a readable JPEG or PNG image.") from exc


def image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
