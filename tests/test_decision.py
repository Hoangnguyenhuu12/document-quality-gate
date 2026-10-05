"""
Unit tests for Readiness Decision Engine.
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas.quality_assessment import PageQualityMetrics
from src.decision.readiness_engine import ReadinessEngine


def test_decision_engine_good_document():
    engine = ReadinessEngine()
    metrics = PageQualityMetrics(
        blur_score=150.0,
        contrast_score=60.0,
        estimated_dpi=250,
        shadow_disparity=10.0
    )
    res = engine.evaluate_page(page_number=1, metrics=metrics)
    assert res.readiness == "good"
    assert res.recommendation == "direct_to_ocr"
    assert res.quality_score >= 3.8
    assert len(res.issues) == 0


def test_decision_engine_blurry_rejection():
    engine = ReadinessEngine()
    metrics = PageQualityMetrics(
        blur_score=15.0,  # Below bad_max (35)
        contrast_score=50.0,
        estimated_dpi=200,
        shadow_disparity=12.0
    )
    res = engine.evaluate_page(page_number=1, metrics=metrics)
    assert res.readiness == "bad"
    assert res.recommendation == "reject"
    assert "severe_blur" in res.issues


def test_decision_engine_medium_enhancement():
    engine = ReadinessEngine()
    metrics = PageQualityMetrics(
        blur_score=60.0,  # Between 35 and 100
        contrast_score=35.0,
        estimated_dpi=180,
        shadow_disparity=25.0
    )
    res = engine.evaluate_page(page_number=1, metrics=metrics)
    assert res.readiness == "medium"
    assert res.recommendation == "enhance_before_ocr"


def test_decision_engine_skew_and_glare_flags():
    engine = ReadinessEngine()
    metrics = PageQualityMetrics(
        blur_score=120.0,
        contrast_score=50.0,
        estimated_dpi=220,
        shadow_disparity=10.0,
        skew_angle=4.5,   # Mild skew
        glare_score=4.0   # Mild glare
    )
    res = engine.evaluate_page(page_number=1, metrics=metrics)
    assert "mild_skew" in res.issues
    assert "mild_glare" in res.issues
    assert res.recommendation == "enhance_before_ocr"

