"""
Central Pipeline for AI Document Quality Gate.
Evaluates documents/images, calculates multi-dimensional physical metrics,
and issues triage decisions (direct_to_ocr, enhance_before_ocr, reject).
"""

from pathlib import Path
from typing import List, Union, Optional
import pymupdf as fitz
import yaml
import hashlib

from ..schemas.quality_assessment import (
    PageQualityMetrics,
    PageAssessment,
    DocumentQualityAssessment,
)
from ..preprocessing.image_loader import QualityImageLoader
from ..preprocessing.image_enhancer import DocumentEnhancer
from ..metrics.blur_detector import BlurDetector
from ..metrics.contrast_detector import ContrastDetector
from ..metrics.resolution_detector import ResolutionDetector
from ..metrics.shadow_detector import ShadowDetector
from ..metrics.skew_detector import SkewDetector
from ..metrics.glare_detector import GlareDetector
from ..metrics.blank_page_detector import BlankPageDetector
from ..metrics.orientation_detector import OrientationDetector
from ..decision.readiness_engine import ReadinessEngine
from ..storage.storage_manager import StorageManager
from ..storage.assessment_cache import AssessmentCache


class DocumentQualityPipeline:
    """Document Quality Gate assessing OCR Readiness and physical defects."""

    def __init__(
        self,
        decision_engine: Optional[ReadinessEngine] = None,
        default_dpi: int = 150,
        storage_manager: Optional[StorageManager] = None,
        cache: Optional[AssessmentCache] = None,
        config_hash: Optional[str] = None
    ):
        self.decision_engine = decision_engine or ReadinessEngine()
        self.default_dpi = default_dpi
        self.image_loader = QualityImageLoader()
        self.enhancer = DocumentEnhancer()
        self.storage = storage_manager or StorageManager()
        self.cache = cache or AssessmentCache()
        self.config_hash = config_hash or "default_cfg"

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path] = "configs/default.yaml") -> "DocumentQualityPipeline":
        """Instantiate pipeline from a YAML configuration file with automatic config hash generation."""
        path = Path(config_path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw_content = f.read()
            cfg = yaml.safe_load(raw_content)

        cfg_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()[:16]

        thresh = cfg.get("thresholds", {})
        blur_cfg = thresh.get("blur", {})
        contrast_cfg = thresh.get("contrast", {})
        res_cfg = thresh.get("resolution", {})
        shadow_cfg = thresh.get("shadow", {})
        skew_cfg = thresh.get("skew", {})
        glare_cfg = thresh.get("glare", {})
        weights = cfg.get("decision", {}).get("weights", {})

        engine = ReadinessEngine(
            blur_bad_max=blur_cfg.get("bad_max", 35.0),
            blur_good_min=blur_cfg.get("good_min", 100.0),
            contrast_bad_max=contrast_cfg.get("bad_max", 20.0),
            contrast_good_min=contrast_cfg.get("good_min", 45.0),
            dpi_bad_max=res_cfg.get("bad_max_dpi", 120),
            dpi_good_min=res_cfg.get("good_min_dpi", 200),
            shadow_bad_min=shadow_cfg.get("bad_min_disparity", 40.0),
            shadow_good_max=shadow_cfg.get("good_max_disparity", 18.0),
            skew_bad_min=skew_cfg.get("bad_min_deg", 7.0),
            skew_good_max=skew_cfg.get("good_max_deg", 1.5),
            glare_bad_min=glare_cfg.get("bad_min_ratio", 10.0),
            glare_good_max=glare_cfg.get("good_max_ratio", 2.0),
            weight_blur=weights.get("blur", 0.30),
            weight_contrast=weights.get("contrast", 0.20),
            weight_dpi=weights.get("resolution", 0.20),
            weight_shadow=weights.get("shadow", 0.10),
            weight_skew=weights.get("skew", 0.10),
            weight_glare=weights.get("glare", 0.10),
        )
        return cls(decision_engine=engine, config_hash=cfg_hash)


    def assess_image(self, image_path: Union[str, Path]) -> DocumentQualityAssessment:
        """Assess quality of a standalone image file (PNG, JPG)."""
        path = Path(image_path)
        rgb_arr = self.image_loader.load_image_as_array(path)
        gray_arr = self.image_loader.to_grayscale(rgb_arr)

        h, w = gray_arr.shape[:2]
        dpi = ResolutionDetector.estimate_dpi(w, h)

        if BlankPageDetector.is_blank_page(gray_arr):
            page_eval = PageAssessment(
                page_number=1,
                page_type="blank_page",
                readiness="good",
                quality_score=5.0,
                issues=["blank_page"],
                metrics=PageQualityMetrics(
                    blur_score=500.0,
                    contrast_score=100.0,
                    estimated_dpi=dpi,
                    shadow_disparity=0.0,
                    skew_angle=0.0,
                    glare_score=0.0
                ),
                recommendation="direct_to_ocr"
            )
        else:
            coarse_rot = OrientationDetector.detect_coarse_rotation(gray_arr)
            metrics = PageQualityMetrics(
                blur_score=BlurDetector.compute_blur_score(gray_arr),
                contrast_score=ContrastDetector.compute_contrast_score(gray_arr),
                estimated_dpi=dpi,
                shadow_disparity=ShadowDetector.compute_shadow_disparity(gray_arr),
                skew_angle=SkewDetector.estimate_skew_angle(gray_arr),
                glare_score=GlareDetector.compute_glare_score(gray_arr),
            )
            page_eval = self.decision_engine.evaluate_page(page_number=1, metrics=metrics)
            page_eval.page_type = "scanned_image"
            if coarse_rot != 0:
                page_eval.issues.append(f"rotated_{coarse_rot}_deg")
                if page_eval.readiness == "good":
                    page_eval.readiness = "medium"
                    page_eval.recommendation = "enhance_before_ocr"

        rec = "reject_and_request_reupload" if page_eval.recommendation == "reject" else page_eval.recommendation

        bad_p = [1] if page_eval.readiness == "bad" else []
        med_p = [1] if page_eval.readiness == "medium" else []

        return DocumentQualityAssessment(
            document_name=path.name,
            total_pages=1,
            composition="all_scanned_images",
            defective_pages=bad_p,
            pages_needing_enhancement=med_p,
            overall_quality=page_eval.quality_score,
            ocr_readiness=page_eval.readiness,
            recommendation=rec,
            issues_summary=page_eval.issues,
            page_assessments=[page_eval]
        )

    def assess_pdf(self, pdf_path: Union[str, Path], max_pages: Optional[int] = None) -> DocumentQualityAssessment:
        """Assess quality of a PDF document page by page with strict per-page criteria."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            doc = fitz.open(path)
        except Exception as e:
            return DocumentQualityAssessment(
                document_name=path.name,
                total_pages=0,
                composition="all_scanned_images",
                defective_pages=[],
                pages_needing_enhancement=[],
                overall_quality=1.0,
                ocr_readiness="bad",
                recommendation="reject_and_request_reupload",
                issues_summary=[f"corrupted_or_unreadable_file: {str(e)}"],
                page_assessments=[]
            )

        if doc.is_encrypted:
            doc.close()
            return DocumentQualityAssessment(
                document_name=path.name,
                total_pages=0,
                composition="all_scanned_images",
                defective_pages=[],
                pages_needing_enhancement=[],
                overall_quality=1.0,
                ocr_readiness="bad",
                recommendation="reject_and_request_reupload",
                issues_summary=["password_protected_document"],
                page_assessments=[]
            )

        total_pages = len(doc)
        pages_to_check = total_pages if max_pages is None else min(total_pages, max_pages)

        evals: List[PageAssessment] = []
        all_issues = set()

        for idx in range(pages_to_check):
            page = doc[idx]
            page_text = page.get_text("text").strip()

            # If page contains substantial native digital text (>= 50 chars), it is born-digital
            if len(page_text) >= 50:
                p_eval = PageAssessment(
                    page_number=idx + 1,
                    page_type="digital_text",
                    readiness="good",
                    quality_score=5.0,
                    issues=[],
                    metrics=PageQualityMetrics(
                        blur_score=500.0,
                        contrast_score=100.0,
                        estimated_dpi=300,
                        shadow_disparity=0.0,
                        skew_angle=0.0,
                        glare_score=0.0
                    ),
                    recommendation="direct_to_ocr"
                )
            else:
                # Scanned image / photograph (e.g. signature page with stamp or scanned contract)
                rgb_arr, dpi = self.image_loader.rasterize_pdf_page(page, dpi=self.default_dpi)
                gray_arr = self.image_loader.to_grayscale(rgb_arr)

                # Check if this is an intentional blank page (e.g. duplex scan)
                if BlankPageDetector.is_blank_page(gray_arr):
                    p_eval = PageAssessment(
                        page_number=idx + 1,
                        page_type="blank_page",
                        readiness="good",
                        quality_score=5.0,
                        issues=["blank_page"],
                        metrics=PageQualityMetrics(
                            blur_score=500.0,
                            contrast_score=100.0,
                            estimated_dpi=dpi,
                            shadow_disparity=0.0,
                            skew_angle=0.0,
                            glare_score=0.0
                        ),
                        recommendation="direct_to_ocr"
                    )
                else:
                    coarse_rot = OrientationDetector.detect_coarse_rotation(gray_arr)
                    metrics = PageQualityMetrics(
                        blur_score=BlurDetector.compute_blur_score(gray_arr),
                        contrast_score=ContrastDetector.compute_contrast_score(gray_arr),
                        estimated_dpi=dpi,
                        shadow_disparity=ShadowDetector.compute_shadow_disparity(gray_arr),
                        skew_angle=SkewDetector.estimate_skew_angle(gray_arr),
                        glare_score=GlareDetector.compute_glare_score(gray_arr),
                    )
                    p_eval = self.decision_engine.evaluate_page(page_number=idx + 1, metrics=metrics)
                    p_eval.page_type = "scanned_image"
                    if coarse_rot != 0:
                        p_eval.issues.append(f"rotated_{coarse_rot}_deg")
                        if p_eval.readiness == "good":
                            p_eval.readiness = "medium"
                            p_eval.recommendation = "enhance_before_ocr"

            evals.append(p_eval)
            for iss in p_eval.issues:
                all_issues.add(iss)

        doc.close()

        if not evals:
            return DocumentQualityAssessment(
                document_name=path.name,
                total_pages=0,
                composition="all_scanned_images",
                defective_pages=[],
                pages_needing_enhancement=[],
                overall_quality=1.0,
                ocr_readiness="bad",
                recommendation="reject_and_request_reupload",
                issues_summary=["empty_document"],
                page_assessments=[]
            )

        avg_score = round(sum(p.quality_score for p in evals) / len(evals), 2)
        bad_pages = [p.page_number for p in evals if p.readiness == "bad"]
        medium_pages = [p.page_number for p in evals if p.readiness == "medium"]

        # Determine document composition
        digital_count = sum(1 for p in evals if p.page_type == "digital_text")
        scanned_count = sum(1 for p in evals if p.page_type in ("scanned_image", "blank_page"))
        if digital_count == len(evals):
            composition = "all_digital_text"
        elif scanned_count == len(evals):
            composition = "all_scanned_images"
        else:
            composition = "mixed_scanned_and_digital"

        # Strict Per-Page Triage Rule:
        # If any single page (e.g. final signature/stamp page) fails badly -> Reject and request reupload
        if bad_pages:
            doc_readiness = "bad"
            doc_recommendation = "reject_and_request_reupload"
            all_issues.add(f"defective_pages:{','.join(map(str, bad_pages))}")
        elif medium_pages:
            doc_readiness = "medium"
            doc_recommendation = "enhance_before_ocr"
            all_issues.add(f"pages_needing_enhancement:{','.join(map(str, medium_pages))}")
        else:
            doc_readiness = "good"
            doc_recommendation = "direct_to_ocr"

        return DocumentQualityAssessment(
            document_name=path.name,
            total_pages=total_pages,
            composition=composition,
            defective_pages=bad_pages,
            pages_needing_enhancement=medium_pages,
            overall_quality=avg_score,
            ocr_readiness=doc_readiness,
            recommendation=doc_recommendation,
            issues_summary=sorted(list(all_issues)),
            page_assessments=evals
        )

    def assess_document(
        self,
        file_path: Union[str, Path],
        max_pages: Optional[int] = None,
        use_cache: bool = True,
        save_report: bool = True
    ) -> DocumentQualityAssessment:
        """
        Route file to image or PDF assessment with transparent SHA-256 caching and persistence.
        By default evaluates ALL pages (max_pages=None) to ensure signatures/stamps at the end are never missed.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_hash = AssessmentCache.compute_file_hash(path)
        file_size = path.stat().st_size

        # 1. Check cache with config_hash for automatic invalidation
        if use_cache:
            cached_result = self.cache.get(file_hash, self.config_hash)
            if cached_result:
                # Remove internal cache keys before schema parsing
                c_clean = {k: v for k, v in cached_result.items() if not k.startswith("_")}
                return DocumentQualityAssessment(**c_clean)

        # 2. Perform physical CV assessment
        suffix = path.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            assessment = self.assess_image(path)
        elif suffix == ".pdf":
            assessment = self.assess_pdf(path, max_pages=max_pages)
        else:
            raise ValueError(f"Unsupported format for quality gate: '{suffix}'")

        # 3. Save report and populate cache
        if save_report:
            self.storage.save_report(assessment, file_hash)

        if use_cache:
            self.cache.set(file_hash, path.name, file_size, assessment, config_hash=self.config_hash)

        return assessment

    def enhance_image_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        skew_angle: Optional[float] = None,
        preserve_color: bool = True
    ) -> Path:
        """
        Enhance a degraded image file and save the clean, OCR-ready result.
        Preserves color (red stamps & blue signatures) by default.
        """
        in_p = Path(input_path)

        rgb_arr = self.image_loader.load_image_as_array(in_p)
        gray_arr = self.image_loader.to_grayscale(rgb_arr)

        if skew_angle is None:
            skew_angle = SkewDetector.estimate_skew_angle(gray_arr)

        coarse_rot = OrientationDetector.detect_coarse_rotation(gray_arr)

        enhanced = self.enhancer.enhance_document_page(
            rgb_arr,
            skew_angle=skew_angle,
            coarse_rotation=coarse_rot,
            apply_deskew=True,
            apply_shadow_removal=True,
            apply_contrast_stretch=True,
            preserve_color=preserve_color
        )

        if output_path is None:
            return self.storage.save_processed_image(enhanced, in_p.name)

        out_p = Path(output_path)
        from PIL import Image
        enhanced_pil = Image.fromarray(enhanced)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        enhanced_pil.save(out_p)
        return out_p

    def enhance_pdf(
        self,
        pdf_path: Union[str, Path],
        output_pdf_path: Optional[Union[str, Path]] = None,
        export_page_images: bool = True,
        preserve_color: bool = True
    ) -> Path:
        """
        Two-Pass "Skim-Scan & Targeted Patch" Mechanism with color preservation & metadata reuse:
        1. Pass 1 (Skim Pass): Rapidly triage all pages to identify defective/skewed pages and record metadata.
        2. Pass 2 (Targeted Patch): Extract ONLY defective pages, reusing Pass 1 skew/orientation metadata.
        3. Pass 3 (In-Place Assembly): Re-assemble the final PDF, preserving 100% of clean native vector pages,
           and patching only enhanced color-preserved pages at their exact original positions.
        """
        import io
        from PIL import Image

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # --- PASS 1: SKIM PASS (Tầm soát nhanh toàn bộ tài liệu & thu thập metadata) ---
        assessment = self.assess_pdf(path, max_pages=None)
        target_pages = set(assessment.defective_pages + assessment.pages_needing_enhancement)
        page_meta_map = {p.page_number: p for p in assessment.page_assessments}

        src_doc = fitz.open(path)
        total_pages = len(src_doc)

        # --- PASS 2: TARGETED ENHANCEMENT (Tái sử dụng metadata từ Pass 1, giữ màu con dấu) ---
        enhanced_patches = {}  # {page_num: (clean_arr, rect)}

        for page_num in target_pages:
            idx = page_num - 1
            if 0 <= idx < total_pages:
                page = src_doc[idx]
                rgb_arr, dpi = self.image_loader.rasterize_pdf_page(page, dpi=self.default_dpi)

                # Reuse precomputed skew and orientation from Pass 1!
                p_eval = page_meta_map.get(page_num)
                skew_angle = p_eval.metrics.skew_angle if p_eval else 0.0
                coarse_rot = 90 if p_eval and any("rotated_90" in iss for iss in p_eval.issues) else 0

                clean_arr = self.enhancer.enhance_document_page(
                    rgb_arr,
                    skew_angle=skew_angle,
                    coarse_rotation=coarse_rot,
                    apply_deskew=True,
                    apply_shadow_removal=True,
                    apply_contrast_stretch=True,
                    preserve_color=preserve_color
                )
                enhanced_patches[page_num] = (clean_arr, page.rect)

        # --- PASS 3: IN-PLACE ASSEMBLY & EXPORT (Ghép nối chính xác theo thứ tự ban đầu) ---
        out_doc = fitz.open()

        for idx in range(total_pages):
            page_num = idx + 1
            page = src_doc[idx]

            if page_num in enhanced_patches:
                # Trang có lỗi: Ghép trang ảnh đã được làm sạch
                clean_arr, rect = enhanced_patches[page_num]
                clean_pil = Image.fromarray(clean_arr)
                img_bytes = io.BytesIO()
                clean_pil.save(img_bytes, format="PNG")

                new_page = out_doc.new_page(width=rect.width, height=rect.height)
                new_page.insert_image(rect, stream=img_bytes.getvalue())

                if export_page_images:
                    self.storage.save_page_image(clean_arr, path.name, page_num=page_num)
            else:
                # Trang đạt chuẩn (Word/vector text hoặc scan đẹp): Copy nguyên bản 100% không suy hao chất lượng
                out_doc.insert_pdf(src_doc, from_page=idx, to_page=idx)

                if export_page_images:
                    rgb_arr, _ = self.image_loader.rasterize_pdf_page(page, dpi=self.default_dpi)
                    self.storage.save_page_image(rgb_arr, path.name, page_num=page_num)

        src_doc.close()

        # Lưu file PDF hoàn chỉnh
        if output_pdf_path is None:
            dest_pdf = self.storage.save_processed_pdf(out_doc, path.name)
        else:
            dest_pdf = Path(output_pdf_path)
            dest_pdf.parent.mkdir(parents=True, exist_ok=True)
            out_doc.save(str(dest_pdf))

        out_doc.close()
        return dest_pdf





