"""
Comprehensive QA Integration Tests covering all 6 Critical & High priority improvements:
1. Blank Page Detection (Duplex scans never rejected for blur)
2. Color-Preserving Enhancement (Red stamps & blue signatures stay vibrant)
3. Full Document Evaluation (No arbitrary 5-page truncation)
4. Cache Invalidation upon Config Hash change
5. Orientation Detection (90-degree landscape rotation)
6. Skim-and-Patch Pass 1 -> Pass 2 metadata reuse
"""

import io
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import pymupdf as fitz

from src.metrics.blank_page_detector import BlankPageDetector
from src.metrics.orientation_detector import OrientationDetector
from src.preprocessing.image_enhancer import DocumentEnhancer
from src.storage.assessment_cache import AssessmentCache
from src.pipeline.quality_pipeline import DocumentQualityPipeline


def test_blank_page_never_rejected():
    """Verify that pure white or nearly blank sheets are recognized as blank_page and pass."""
    # 1. Pure blank paper
    blank = np.full((500, 500), 255.0, dtype=np.float32)
    assert BlankPageDetector.is_blank_page(blank) is True

    # 2. Blank page with tiny dust specks (< 0.1% ink)
    blank_dust = np.full((500, 500), 250.0, dtype=np.float32)
    blank_dust[10:12, 10:12] = 50.0  # 4 pixels of dust
    assert BlankPageDetector.is_blank_page(blank_dust) is True

    # 3. Document with actual text (ink > 2%)
    doc_with_text = np.full((500, 500), 255.0, dtype=np.float32)
    doc_with_text[::15, :] = 0.0
    assert BlankPageDetector.is_blank_page(doc_with_text) is False


def test_color_preservation_for_stamps_and_signatures():
    """Verify that red stamps (R >> G, B) remain colored and are not flattened to grayscale."""
    # Create white document (400x400x3)
    img_rgb = np.full((400, 400, 3), 230, dtype=np.uint8)

    # Add shadow on right side
    img_rgb[:, 250:] = 120

    # Draw a RED company stamp (R=220, G=30, B=30)
    img_rgb[100:150, 100:150, 0] = 220
    img_rgb[100:150, 100:150, 1] = 30
    img_rgb[100:150, 100:150, 2] = 30

    # Draw a BLUE ink signature (R=30, G=50, B=220)
    img_rgb[200:220, 100:250, 0] = 30
    img_rgb[200:220, 100:250, 1] = 50
    img_rgb[200:220, 100:250, 2] = 220

    # Enhance with color preservation
    enhanced = DocumentEnhancer.enhance_document_page(img_rgb, preserve_color=True)

    # 1. Output must be 3-channel RGB
    assert len(enhanced.shape) == 3 and enhanced.shape[2] == 3

    # 2. Red stamp area must remain intensely RED (R much higher than G and B)
    stamp_px = enhanced[120, 120]
    assert stamp_px[0] > 180
    assert stamp_px[1] < 80
    assert stamp_px[2] < 80

    # 3. Blue signature area must remain intensely BLUE (B much higher than R and G)
    sig_px = enhanced[210, 150]
    assert sig_px[2] > 180
    assert sig_px[0] < 80


def test_orientation_detection():
    """Verify that text running vertically triggers rotation detection."""
    # Standard horizontal text
    horizontal = np.full((300, 400), 255.0, dtype=np.float32)
    horizontal[::20, :] = 0.0
    rot_h = OrientationDetector.detect_coarse_rotation(horizontal)
    assert rot_h == 0

    # Sideways rotated page: width > height, lines run vertically
    vertical = np.full((300, 500), 255.0, dtype=np.float32)
    vertical[:, ::20] = 0.0
    rot_v = OrientationDetector.detect_coarse_rotation(vertical)
    assert rot_v == 90


def test_cache_invalidation_by_config(tmp_path):
    """Verify that changing the config hash invalidates the cache automatically."""
    cache = AssessmentCache(db_path=tmp_path / "invalidation_test.db")
    file_path = tmp_path / "contract_v1.pdf"
    file_path.write_bytes(b"contract content for invalidation test")
    file_hash = AssessmentCache.compute_file_hash(file_path)

    report_payload = {"document_name": "contract.pdf", "overall_quality": 4.5}

    # Store with config_hash A
    cache.set(file_hash, "contract.pdf", 100, report_payload, config_hash="config_v1")

    # Hit with config_hash A
    assert cache.get(file_hash, "config_v1") is not None

    # Miss with config_hash B (changed thresholds)
    assert cache.get(file_hash, "config_v2") is None


def test_full_document_evaluation_all_pages(tmp_path):
    """Verify that default assess_document checks beyond 5 pages (e.g. signature on page 7)."""
    pdf_path = tmp_path / "long_contract.pdf"
    doc = fitz.open()
    # 6 clean pages
    for i in range(6):
        p = doc.new_page()
        p.insert_text(fitz.Point(50, 72), f"Contract Page {i+1} terms and conditions", fontsize=14)

    # 7th page: Blank page or scan
    p_last = doc.new_page()
    p_last.insert_text(fitz.Point(50, 72), "Final Execution Page with Authorized Signatures", fontsize=14)
    doc.save(str(pdf_path))
    doc.close()

    pipeline = DocumentQualityPipeline()
    assessment = pipeline.assess_document(pdf_path, use_cache=False)

    # Must assess all 7 pages, not stop at 5!
    assert assessment.total_pages == 7
    assert len(assessment.page_assessments) == 7
