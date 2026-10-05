"""
Automatic Document Image Enhancement Module.
Transforms degraded/skewed/shadowed pages into clean, OCR-ready images.
"""

from typing import Union
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter


class DocumentEnhancer:
    """Enhances degraded document images (deskew, shadow removal, contrast stretch)."""

    @staticmethod
    def deskew(image: Union[np.ndarray, Image.Image], angle: float) -> np.ndarray:
        """Rotate image to correct tilt angle (in degrees)."""
        if abs(angle) < 0.2:
            if isinstance(image, Image.Image):
                return np.array(image, dtype=np.uint8)
            return image

        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8))
        else:
            pil_img = image

        # Rotate with white background fill
        bg_color = (255, 255, 255) if pil_img.mode == "RGB" else 255
        rotated = pil_img.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=bg_color)
        return np.array(rotated, dtype=np.uint8)

    @staticmethod
    def remove_shadows_and_flatten_lighting(
        image_array: np.ndarray,
        blur_radius: int = 25,
        preserve_color: bool = True
    ) -> np.ndarray:
        """
        Estimate background lighting map via large-kernel Gaussian filter
        and divide/normalize the image to eliminate corner shadows and folds.
        When preserve_color is True and image is RGB, scales each color channel
        so white paper is normalized while preserving red stamps and blue ink signatures.
        """
        is_rgb = (len(image_array.shape) == 3 and image_array.shape[2] == 3)

        if is_rgb:
            # Luminance for illumination estimation
            gray = np.dot(image_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray = image_array.astype(np.uint8)

        pil_gray = Image.fromarray(gray, mode="L")
        bg = pil_gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        bg_arr = np.maximum(np.array(bg, dtype=np.float32), 1.0)

        gain = 255.0 / bg_arr

        if is_rgb and preserve_color:
            cleaned = np.zeros_like(image_array, dtype=np.uint8)
            for c in range(3):
                ch = image_array[..., c].astype(np.float32) * gain
                cleaned[..., c] = np.clip(ch, 0, 255).astype(np.uint8)
            return cleaned
        else:
            normalized = (gray.astype(np.float32) / bg_arr) * 255.0
            return np.clip(normalized, 0, 255).astype(np.uint8)

    @staticmethod
    def stretch_contrast(image_array: np.ndarray, low_percentile: float = 2.0, high_percentile: float = 98.0) -> np.ndarray:
        """
        Normalize dynamic range so faint ink becomes crisp black and off-white paper becomes pure white.
        Supports both grayscale and RGB channels.
        """
        is_rgb = (len(image_array.shape) == 3 and image_array.shape[2] == 3)

        if is_rgb:
            gray = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
            p_low = np.percentile(gray, low_percentile)
            p_high = np.percentile(gray, high_percentile)
            if p_high <= p_low:
                return image_array
            scale = 255.0 / (p_high - p_low)
            stretched = (image_array.astype(np.float32) - p_low) * scale
            return np.clip(stretched, 0, 255).astype(np.uint8)
        else:
            p_low = np.percentile(image_array, low_percentile)
            p_high = np.percentile(image_array, high_percentile)
            if p_high <= p_low:
                return image_array
            stretched = (image_array - p_low) / (p_high - p_low) * 255.0
            return np.clip(stretched, 0, 255).astype(np.uint8)

    @classmethod
    def enhance_document_page(
        cls,
        rgb_or_gray: np.ndarray,
        skew_angle: float = 0.0,
        coarse_rotation: int = 0,
        apply_deskew: bool = True,
        apply_shadow_removal: bool = True,
        apply_contrast_stretch: bool = True,
        preserve_color: bool = True
    ) -> np.ndarray:
        """
        Full enhancement pipeline for a document page.
        Supports color preservation (keeps red company stamps & blue signatures crisp).
        """
        img = rgb_or_gray

        # 1. Coarse Rotation (90, 180, 270 deg)
        if coarse_rotation in (90, 180, 270):
            pil_img = Image.fromarray(img)
            img = np.array(pil_img.rotate(-coarse_rotation, expand=True))

        # 2. Fine Deskew
        if apply_deskew and abs(skew_angle) >= 0.5:
            img = cls.deskew(img, skew_angle)

        # 3. Shadow Removal (Preserving red stamps and blue signatures if RGB)
        if apply_shadow_removal:
            img = cls.remove_shadows_and_flatten_lighting(img, preserve_color=preserve_color)

        # 4. Contrast Stretch
        if apply_contrast_stretch:
            img = cls.stretch_contrast(img)

        return img

