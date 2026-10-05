"""
Integration tests for DocumentQualityPipeline.
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import pymupdf as fitz
try:
    import pytest
    fixture = pytest.fixture
except ImportError:
    def fixture(fn):
        return fn

from src.pipeline.quality_pipeline import DocumentQualityPipeline


@fixture

def sample_pdf(tmp_path) -> Path:
    pdf_path = tmp_path / "sample_doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(50, 72), "Official Company Policy Document 2026", fontsize=16)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@fixture
def sample_image(tmp_path) -> Path:
    img_path = tmp_path / "scanned_photo.png"
    # Create 300x300 image
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[50:250, 50:250] = 255  # White document center on black background
    img = Image.fromarray(arr)
    img.save(img_path)
    return img_path


def test_assess_pdf(sample_pdf=None):
    if sample_pdf is None or (isinstance(sample_pdf, Path) and sample_pdf.is_dir()):
        import tempfile
        tmp = sample_pdf if (isinstance(sample_pdf, Path) and sample_pdf.is_dir()) else Path(tempfile.mkdtemp())
        sample_pdf = (tmp / "sample_doc.pdf")
        doc = fitz.open()
        p = doc.new_page()
        p.insert_text(fitz.Point(50, 72), "Official Company Policy Document 2026", fontsize=16)
        doc.save(sample_pdf)
        doc.close()

    pipeline = DocumentQualityPipeline()
    assessment = pipeline.assess_document(sample_pdf)

    assert assessment.document_name == "sample_doc.pdf"
    assert assessment.total_pages == 1
    assert assessment.overall_quality >= 1.0
    assert assessment.ocr_readiness in ("good", "medium", "bad")
    assert len(assessment.page_assessments) == 1


def test_assess_image(sample_image=None):
    if sample_image is None or (isinstance(sample_image, Path) and sample_image.is_dir()):
        import tempfile
        tmp = sample_image if (isinstance(sample_image, Path) and sample_image.is_dir()) else Path(tempfile.mkdtemp())
        sample_image = (tmp / "scanned_photo.png")
        arr = np.zeros((300, 300, 3), dtype=np.uint8)
        arr[50:250, 50:250] = 255
        Image.fromarray(arr).save(sample_image)

    pipeline = DocumentQualityPipeline()
    assessment = pipeline.assess_document(sample_image)

    assert assessment.document_name == "scanned_photo.png"
    assert assessment.total_pages == 1
    assert len(assessment.page_assessments) == 1


def test_enhance_image(sample_image=None, tmp_path=None):
    if sample_image is None or (isinstance(sample_image, Path) and sample_image.is_dir()):
        import tempfile
        tmp = sample_image if (isinstance(sample_image, Path) and sample_image.is_dir()) else Path(tempfile.mkdtemp())
        sample_image = (tmp / "scanned_photo.png")
        arr = np.zeros((300, 300, 3), dtype=np.uint8)
        arr[50:250, 50:250] = 255
        Image.fromarray(arr).save(sample_image)
        tmp_path = tmp
    elif tmp_path is None:
        import tempfile
        tmp_path = Path(tempfile.mkdtemp())

    pipeline = DocumentQualityPipeline()
    out_path = tmp_path / "enhanced_result.png"
    enhanced_file = pipeline.enhance_image_file(sample_image, out_path, skew_angle=2.0)
    assert enhanced_file.exists()
    assert enhanced_file.stat().st_size > 0

