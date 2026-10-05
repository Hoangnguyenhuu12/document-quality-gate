"""
Command-Line Interface (CLI) for AI Document Quality Gate.
Usage:
    python cli.py evaluate sample.pdf
    python cli.py evaluate scanned_page.png --output quality_result.json
"""

import argparse
import json
import sys
from pathlib import Path

# Configure utf-8 encoding for stdout on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.quality_pipeline import DocumentQualityPipeline


def main():
    parser = argparse.ArgumentParser(description="AI Document Quality Gate CLI")
    parser.add_argument("file_path", type=str, help="Path to document (PDF, PNG, JPG)")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--output", "-o", type=str, default="", help="Path to output JSON")
    parser.add_argument("--enhance", "-e", type=str, default="", help="Path to save enhanced image (for PNG/JPG)")
    parser.add_argument("--no-cache", action="store_true", help="Bypass SHA-256 cache and force re-analysis")

    args = parser.parse_args()

    input_path = Path(args.file_path)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    cfg_path = Path(__file__).parent / args.config
    pipeline = DocumentQualityPipeline.from_config_file(cfg_path)

    print(f"Assessing document quality for '{input_path.name}'...")
    assessment = pipeline.assess_document(input_path, use_cache=not args.no_cache, save_report=True)

    json_str = json.dumps(assessment.model_dump(), ensure_ascii=False, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json_str, encoding="utf-8")
        print(f"Saved assessment to: {out_path}")
    else:
        print("\n--- QUALITY GATE ASSESSMENT ---")
        print(f"Document: {assessment.document_name}")
        print(f"Overall Quality Score: {assessment.overall_quality}/5.0")
        print(f"OCR Readiness: {assessment.ocr_readiness.upper()}")
        print(f"Recommendation: {assessment.recommendation.upper()}")
        if assessment.issues_summary:
            print(f"Issues Detected: {', '.join(assessment.issues_summary)}")
        else:
            print("Issues Detected: None (Clean document)")
        print(f"Total Pages Analyzed: {assessment.total_pages}")
        print(f"Document Composition: {assessment.composition.upper()}")
        if assessment.defective_pages:
            print(f"Defective Pages (REJECT): {assessment.defective_pages}")
        if assessment.pages_needing_enhancement:
            print(f"Pages Needing Enhancement: {assessment.pages_needing_enhancement}")
        if assessment.page_assessments:
            first_m = assessment.page_assessments[0].metrics
            print(f"Key Metrics (Page 1): Type={assessment.page_assessments[0].page_type}, Blur={first_m.blur_score}, Contrast={first_m.contrast_score}, DPI={first_m.estimated_dpi}, Shadow={first_m.shadow_disparity}, Skew={first_m.skew_angle}°, Glare={first_m.glare_score}%")
        print(f"\n[Storage Info] Report saved in: storage/reports/")

    if args.enhance:
        suffix = input_path.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            enhanced_path = pipeline.enhance_image_file(input_path, args.enhance)
            print(f"Auto-enhanced image saved to: {enhanced_path}")
        elif suffix == ".pdf":
            out_pdf = args.enhance if args.enhance.lower().endswith(".pdf") else None
            enhanced_path = pipeline.enhance_pdf(input_path, output_pdf_path=out_pdf, export_page_images=True)
            print(f"Auto-enhanced PDF saved to: {enhanced_path}")
            print(f"Ordered clean page images exported to: storage/processed/{input_path.stem}/")
        else:
            print(f"Warning: Unsupported format for enhance: {suffix}")


if __name__ == "__main__":

    main()

