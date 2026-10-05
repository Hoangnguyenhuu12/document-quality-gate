"""
Standardized Quality Assessment Schemas for AI Document Quality Gate.
Encapsulates OCR Readiness metrics, detected defects, and routing decisions.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PageQualityMetrics(BaseModel):
    """Raw computed computer-vision metrics for a document page."""
    blur_score: float = Field(description="Laplacian energy variance; higher is sharper")
    contrast_score: float = Field(description="Luminance standard deviation")
    estimated_dpi: int = Field(description="Estimated DPI against standard A4 geometry")
    shadow_disparity: float = Field(description="Illumination disparity across page quadrants")
    skew_angle: float = Field(default=0.0, description="Estimated page tilt/skew angle in degrees")
    glare_score: float = Field(default=0.0, description="Percentage of severely overexposed/saturated pixels")



class PageAssessment(BaseModel):
    """Assessment and recommendation for a single page."""
    page_number: int
    page_type: str = Field(default="scanned_image", description="'digital_text' or 'scanned_image'")
    readiness: str = Field(description="'good', 'medium', or 'bad'")
    quality_score: float = Field(description="Calibrated score from 1.0 (worst) to 5.0 (best)")
    issues: List[str] = Field(default_factory=list, description="List of detected defects (e.g. blur, shadow)")
    metrics: PageQualityMetrics
    recommendation: str = Field(description="'direct_to_ocr', 'enhance_before_ocr', or 'reject'")


class DocumentQualityAssessment(BaseModel):
    """Overall document triage decision and per-page evaluations."""
    document_name: str
    total_pages: int
    composition: str = Field(default="all_scanned_images", description="'all_scanned_images', 'all_digital_text', or 'mixed_scanned_and_digital'")
    defective_pages: List[int] = Field(default_factory=list, description="List of page numbers failing quality gate")
    pages_needing_enhancement: List[int] = Field(default_factory=list, description="List of page numbers requiring enhancement")
    overall_quality: float = Field(description="Average quality score (1.0 to 5.0)")
    ocr_readiness: str = Field(description="'good', 'medium', or 'bad'")
    recommendation: str = Field(description="'direct_to_ocr', 'enhance_before_ocr', or 'reject_and_request_reupload'")
    issues_summary: List[str] = Field(default_factory=list)
    page_assessments: List[PageAssessment] = Field(default_factory=list)

