"""
Tests for Mixed PDF handling (scanned image pages + digital text pages) and PDF enhancement.
"""

import io
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import pymupdf as fitz

from src.pipeline.quality_pipeline import DocumentQualityPipeline


def create_mixed_contract_pdf(dest_path: Path) -> Path:
    """
    Creates a 3-page mixed contract PDF:
    - Page 1: Scanned image (photograph with dark corner)
    - Page 2: Scanned image (tilted text)
    - Page 3: Born-digital text (clean policy text from Word)
    """
    doc = fitz.open()

    # Page 1: Scanned photo with shadow
    p1 = doc.new_page(width=595, height=842)
    arr1 = np.full((600, 400), 220, dtype=np.uint8)
    arr1[400:, 200:] = 50  # Shadow corner
    pil1 = Image.fromarray(arr1)
    buf1 = io.BytesIO()
    pil1.save(buf1, format="PNG")
    p1.insert_image(p1.rect, stream=buf1.getvalue())

    # Page 2: Scanned photo with tilt
    p2 = doc.new_page(width=595, height=842)
    arr2 = np.full((600, 400), 230, dtype=np.uint8)
    arr2[::25, :] = 20  # Lines of text
    pil2 = Image.fromarray(arr2).rotate(3.5, fillcolor=255)
    buf2 = io.BytesIO()
    pil2.save(buf2, format="PNG")
    p2.insert_image(p2.rect, stream=buf2.getvalue())

    # Page 3: Born-digital text page
    p3 = doc.new_page(width=595, height=842)
    p3.insert_text(
        fitz.Point(50, 72),
        "DIEU 15: DIEU KHOAN THI HANH VA CAM KET BAO MAT THONG TIN\n"
        "Hop dong nay co hieu luc ke tu ngay ky va dong dau cua ca hai ben.\n"
        "Moi tranh chap phat sinh se duoc giai quyet tai Toa an Nhan dan.",
        fontsize=13
    )

    doc.save(str(dest_path))
    doc.close()
    return dest_path


def test_mixed_pdf_assessment(tmp_path):
    pdf_path = tmp_path / "hop_dong_hon_hop.pdf"
    create_mixed_contract_pdf(pdf_path)

    pipeline = DocumentQualityPipeline()
    assessment = pipeline.assess_document(pdf_path, use_cache=False)

    # 1. Total pages
    assert assessment.total_pages == 3

    # 2. Composition detection
    assert assessment.composition == "mixed_scanned_and_digital"

    # 3. Page 1 & 2 are scanned images, Page 3 is digital text
    assert assessment.page_assessments[0].page_type == "scanned_image"
    assert assessment.page_assessments[1].page_type == "scanned_image"
    assert assessment.page_assessments[2].page_type == "digital_text"
    assert assessment.page_assessments[2].readiness == "good"

    # 4. Strict check
    assert len(assessment.defective_pages) > 0 or len(assessment.pages_needing_enhancement) > 0


def test_mixed_pdf_enhancement(tmp_path):
    pdf_path = tmp_path / "hop_dong_can_sua.pdf"
    create_mixed_contract_pdf(pdf_path)

    pipeline = DocumentQualityPipeline()
    out_pdf = tmp_path / "hop_dong_da_sua.pdf"

    enhanced_file = pipeline.enhance_pdf(pdf_path, output_pdf_path=out_pdf, export_page_images=True)
    assert enhanced_file.exists()
    assert enhanced_file.stat().st_size > 0

    # Verify that the enhanced PDF has the exact same page count (3 pages) in exact order
    check_doc = fitz.open(enhanced_file)
    assert len(check_doc) == 3

    # Page 3 must still retain the original digital text layer
    text_p3 = check_doc[2].get_text("text")
    assert "DIEU 15: DIEU KHOAN THI HANH" in text_p3
    check_doc.close()

    # Verify ordered page images exported in storage/processed/<doc_stem>/
    doc_folder = pipeline.storage.processed_dir / pdf_path.stem
    assert doc_folder.exists()
    assert (doc_folder / "page_001.png").exists()
    assert (doc_folder / "page_002.png").exists()
    assert (doc_folder / "page_003.png").exists()
