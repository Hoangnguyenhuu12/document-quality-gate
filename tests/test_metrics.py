"""
Unit tests for physical quality detection metrics.
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.metrics.blur_detector import BlurDetector
from src.metrics.contrast_detector import ContrastDetector
from src.metrics.resolution_detector import ResolutionDetector
from src.metrics.shadow_detector import ShadowDetector


def test_blur_detector_sharp_vs_flat():
    # 1. High-frequency checkerboard pattern -> high Laplacian variance
    sharp = np.zeros((100, 100), dtype=np.float32)
    sharp[::2, ::2] = 255.0
    sharp_score = BlurDetector.compute_blur_score(sharp)

    # 2. Completely flat / smooth image -> 0 Laplacian variance
    flat = np.full((100, 100), 128.0, dtype=np.float32)
    flat_score = BlurDetector.compute_blur_score(flat)

    assert sharp_score > 500.0
    assert flat_score == 0.0


def test_contrast_detector():
    # High contrast: half black, half white
    high_c = np.zeros((100, 100), dtype=np.float32)
    high_c[:, 50:] = 255.0
    score_high = ContrastDetector.compute_contrast_score(high_c)

    # Low contrast: minor gray variations
    low_c = np.full((100, 100), 128.0, dtype=np.float32)
    low_c[0, 0] = 130.0
    score_low = ContrastDetector.compute_contrast_score(low_c)

    assert score_high > 100.0
    assert score_low < 5.0


def test_resolution_detector():
    # Standard 300 DPI A4 is approx 2480 x 3508
    dpi_high = ResolutionDetector.estimate_dpi(2480, 3508)
    assert 280 <= dpi_high <= 320

    # Low resolution thumbnail 400 x 600
    dpi_low = ResolutionDetector.estimate_dpi(400, 600)
    assert dpi_low <= 60


def test_shadow_detector():
    # Uniform illumination across quadrants
    uniform = np.full((120, 120), 200.0, dtype=np.float32)
    disp_uniform = ShadowDetector.compute_shadow_disparity(uniform)
    assert disp_uniform == 0.0

    # Heavy corner shadow: bottom right is dark
    shadowed = np.full((120, 120), 220.0, dtype=np.float32)
    shadowed[60:, 60:] = 40.0
    disp_shadow = ShadowDetector.compute_shadow_disparity(shadowed)
    assert disp_shadow > 40.0


def test_skew_detector_horizontal():
    from src.metrics.skew_detector import SkewDetector
    # Generate horizontal lines mimicking contract text lines
    img = np.full((200, 200), 255.0, dtype=np.float32)
    img[::10, :] = 0.0  # Horizontal text lines every 10 pixels
    angle = SkewDetector.estimate_skew_angle(img)
    assert abs(angle) <= 1.0


def test_glare_detector():
    from src.metrics.glare_detector import GlareDetector
    # Normal document: moderate brightness
    clean = np.full((100, 100), 220.0, dtype=np.float32)
    score_clean = GlareDetector.compute_glare_score(clean)
    assert score_clean == 0.0

    # Flash hotspot: 20% area saturated at 255
    hotspot = np.full((100, 100), 220.0, dtype=np.float32)
    hotspot[:20, :] = 255.0
    score_glare = GlareDetector.compute_glare_score(hotspot)
    assert score_glare >= 20.0

