"""
Unit tests for StorageManager and AssessmentCache.
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.storage.storage_manager import StorageManager
from src.storage.assessment_cache import AssessmentCache
from src.schemas.quality_assessment import (
    DocumentQualityAssessment,
    PageAssessment,
    PageQualityMetrics,
)


def test_storage_manager(tmp_path):
    storage = StorageManager(base_dir=tmp_path / "custom_storage")

    # 1. Test save upload
    raw_file = tmp_path / "raw.pdf"
    raw_file.write_bytes(b"%PDF-1.4 dummy content")
    saved_upload = storage.save_upload(raw_file, "contract_2026.pdf")
    assert saved_upload.exists()
    assert "uploads" in str(saved_upload)

    # 2. Test save processed image
    img_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    saved_img = storage.save_processed_image(img_arr, "contract_page_1.png")
    assert saved_img.exists()
    assert "processed" in str(saved_img)

    # 3. Test save and get report
    report_data = {"document_name": "contract.pdf", "overall_quality": 4.5}
    saved_rep = storage.save_report(report_data, "hash_123")
    assert saved_rep.exists()

    retrieved = storage.get_report("hash_123")
    assert retrieved is not None
    assert retrieved["document_name"] == "contract.pdf"


def test_assessment_cache(tmp_path):
    db_file = tmp_path / "test_cache.db"
    cache = AssessmentCache(db_path=db_file)

    sample_file = tmp_path / "sample.pdf"
    sample_file.write_bytes(b"content for hashing")
    file_hash = AssessmentCache.compute_file_hash(sample_file)

    # Initially empty
    assert cache.get(file_hash) is None

    # Populate cache
    metrics = PageQualityMetrics(
        blur_score=100.0,
        contrast_score=50.0,
        estimated_dpi=200,
        shadow_disparity=10.0,
        skew_angle=0.0,
        glare_score=0.0,
    )
    assessment = DocumentQualityAssessment(
        document_name="sample.pdf",
        total_pages=1,
        overall_quality=4.8,
        ocr_readiness="good",
        recommendation="direct_to_ocr",
        issues_summary=[],
        page_assessments=[
            PageAssessment(
                page_number=1,
                readiness="good",
                quality_score=4.8,
                issues=[],
                metrics=metrics,
                recommendation="direct_to_ocr",
            )
        ],
    )

    cache.set(file_hash, "sample.pdf", sample_file.stat().st_size, assessment)

    # Retrieve from cache
    cached = cache.get(file_hash)
    assert cached is not None
    assert cached["document_name"] == "sample.pdf"
    assert cached["_cached"] is True
    assert cached["overall_quality"] == 4.8
