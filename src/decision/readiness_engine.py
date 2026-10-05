"""
Readiness Decision Engine for AI Document Quality Gate.
Evaluates OCR Readiness, synthesizes multi-metric indicators into quality score,
flags specific issues, and routes to direct_to_ocr / enhance_before_ocr / reject.
"""

from typing import List, Dict, Any, Tuple
from ..schemas.quality_assessment import PageQualityMetrics, PageAssessment


class ReadinessEngine:
    """Decision logic evaluating whether a document page is ready for OCR."""

    def __init__(
        self,
        blur_bad_max: float = 35.0,
        blur_good_min: float = 100.0,
        contrast_bad_max: float = 20.0,
        contrast_good_min: float = 45.0,
        dpi_bad_max: int = 120,
        dpi_good_min: int = 200,
        shadow_bad_min: float = 40.0,
        shadow_good_max: float = 18.0,
        skew_bad_min: float = 7.0,
        skew_good_max: float = 1.5,
        glare_bad_min: float = 10.0,
        glare_good_max: float = 2.0,
        weight_blur: float = 0.30,
        weight_contrast: float = 0.20,
        weight_dpi: float = 0.20,
        weight_shadow: float = 0.10,
        weight_skew: float = 0.10,
        weight_glare: float = 0.10,
    ):
        self.blur_bad_max = blur_bad_max
        self.blur_good_min = blur_good_min
        self.contrast_bad_max = contrast_bad_max
        self.contrast_good_min = contrast_good_min
        self.dpi_bad_max = dpi_bad_max
        self.dpi_good_min = dpi_good_min
        self.shadow_bad_min = shadow_bad_min
        self.shadow_good_max = shadow_good_max
        self.skew_bad_min = skew_bad_min
        self.skew_good_max = skew_good_max
        self.glare_bad_min = glare_bad_min
        self.glare_good_max = glare_good_max

        self.w_blur = weight_blur
        self.w_contrast = weight_contrast
        self.w_dpi = weight_dpi
        self.w_shadow = weight_shadow
        self.w_skew = weight_skew
        self.w_glare = weight_glare

    def _normalize_subscore(self, value: float, min_val: float, max_val: float, invert: bool = False) -> float:
        """Map value to 1.0 - 5.0 score range."""
        if max_val == min_val:
            return 3.0
        ratio = (value - min_val) / (max_val - min_val)
        ratio = max(0.0, min(1.0, ratio))
        if invert:
            ratio = 1.0 - ratio
        return round(1.0 + ratio * 4.0, 2)

    def evaluate_page(self, page_number: int, metrics: PageQualityMetrics) -> PageAssessment:
        """Synthesize metrics and determine readiness and recommendation."""
        issues: List[str] = []

        # 1. Subscores
        s_blur = self._normalize_subscore(metrics.blur_score, self.blur_bad_max, self.blur_good_min)
        if metrics.blur_score < self.blur_bad_max:
            issues.append("severe_blur")
        elif metrics.blur_score < self.blur_good_min:
            issues.append("mild_blur")

        s_contrast = self._normalize_subscore(metrics.contrast_score, self.contrast_bad_max, self.contrast_good_min)
        if metrics.contrast_score < self.contrast_bad_max:
            issues.append("low_contrast")

        s_dpi = self._normalize_subscore(float(metrics.estimated_dpi), float(self.dpi_bad_max), float(self.dpi_good_min))
        if metrics.estimated_dpi < self.dpi_bad_max:
            issues.append("low_resolution")

        s_shadow = self._normalize_subscore(metrics.shadow_disparity, self.shadow_good_max, self.shadow_bad_min, invert=True)
        if metrics.shadow_disparity > self.shadow_bad_min:
            issues.append("harsh_shadow")

        abs_skew = abs(metrics.skew_angle)
        s_skew = self._normalize_subscore(abs_skew, self.skew_good_max, self.skew_bad_min, invert=True)
        if abs_skew > self.skew_bad_min:
            issues.append("severe_skew")
        elif abs_skew > self.skew_good_max:
            issues.append("mild_skew")

        s_glare = self._normalize_subscore(metrics.glare_score, self.glare_good_max, self.glare_bad_min, invert=True)
        if metrics.glare_score > self.glare_bad_min:
            issues.append("harsh_glare")
        elif metrics.glare_score > self.glare_good_max:
            issues.append("mild_glare")

        # 2. Weighted overall quality score (1.0 to 5.0)
        overall_score = round(
            self.w_blur * s_blur +
            self.w_contrast * s_contrast +
            self.w_dpi * s_dpi +
            self.w_shadow * s_shadow +
            self.w_skew * s_skew +
            self.w_glare * s_glare,
            2
        )

        # 3. Categorize readiness & recommendation
        # Hard fail rule: If severely blurry, severely low resolution, or blinded by glare -> Reject directly
        if "severe_blur" in issues or metrics.estimated_dpi < 90 or metrics.glare_score >= 25.0 or overall_score < 2.5:
            readiness = "bad"
            recommendation = "reject"
        elif overall_score >= 3.8 and not issues:
            readiness = "good"
            recommendation = "direct_to_ocr"
        else:
            readiness = "medium"
            recommendation = "enhance_before_ocr"

        return PageAssessment(
            page_number=page_number,
            readiness=readiness,
            quality_score=overall_score,
            issues=issues,
            metrics=metrics,
            recommendation=recommendation
        )

