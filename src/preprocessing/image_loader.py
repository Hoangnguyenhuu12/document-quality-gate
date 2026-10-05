"""
Image and Document Page Ingestion for Quality Analysis.
Loads image files (PNG, JPG) or rasterizes PDF pages into NumPy arrays.
"""

from pathlib import Path
from typing import List, Union, Tuple
import numpy as np
from PIL import Image
import pymupdf as fitz


class QualityImageLoader:
    """Loads pages and images into standard grayscale/RGB numpy matrices."""

    @staticmethod
    def load_image_as_array(image_path: Union[str, Path]) -> np.ndarray:
        """Load image file from disk as uint8 numpy array."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)

    @staticmethod
    def rasterize_pdf_page(page: fitz.Page, dpi: int = 150) -> Tuple[np.ndarray, int]:
        """
        Rasterize a fitz PDF page to uint8 RGB array at specified DPI.
        Returns (image_array, estimated_dpi).
        """
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return np.array(img, dtype=np.uint8), dpi

    @staticmethod
    def to_grayscale(img_array: np.ndarray) -> np.ndarray:
        """Convert RGB image array to grayscale using standard ITU-R luminance weights."""
        if len(img_array.shape) == 2:
            return img_array
        # Y = 0.299 R + 0.587 G + 0.114 B
        gray = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
        return gray.astype(np.float32)
