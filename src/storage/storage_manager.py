"""
Storage Manager for AI Document Quality Gate.
Handles structured local persistence for original uploads, enhanced/cleaned images,
quality assessment JSON reports, and cache databases.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any
import json
import shutil
from datetime import datetime
from PIL import Image
import numpy as np


class StorageManager:
    """Manages workspace storage directories and persistent artifacts."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        if base_dir is None:
            # Default to <project_root>/storage
            self.base_dir = Path(__file__).resolve().parent.parent.parent / "storage"
        else:
            self.base_dir = Path(base_dir)

        self.uploads_dir = self.base_dir / "uploads"
        self.processed_dir = self.base_dir / "processed"
        self.reports_dir = self.base_dir / "reports"
        self.cache_dir = self.base_dir / "cache"

        # Ensure all required directories exist
        for d in (self.uploads_dir, self.processed_dir, self.reports_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save_upload(self, source_path_or_bytes: Union[str, Path, bytes], filename: str) -> Path:
        """Store an uploaded file into storage/uploads/ with date subfolders."""
        date_folder = self.uploads_dir / datetime.now().strftime("%Y-%m-%d")
        date_folder.mkdir(parents=True, exist_ok=True)
        dest_path = date_folder / filename

        if isinstance(source_path_or_bytes, bytes):
            dest_path.write_bytes(source_path_or_bytes)
        else:
            src = Path(source_path_or_bytes)
            if src.resolve() != dest_path.resolve():
                shutil.copy2(src, dest_path)
        return dest_path

    def save_processed_image(self, image_data: Union[np.ndarray, Image.Image], base_name: str, suffix: str = ".png") -> Path:
        """Save cleaned/enhanced image to storage/processed/."""
        clean_stem = Path(base_name).stem
        dest_path = self.processed_dir / f"enhanced_{clean_stem}{suffix}"

        if isinstance(image_data, np.ndarray):
            pil_img = Image.fromarray(image_data.astype(np.uint8))
        else:
            pil_img = image_data

        pil_img.save(dest_path)
        return dest_path

    def save_page_image(
        self,
        image_data: Union[np.ndarray, Image.Image],
        document_name: str,
        page_num: int,
        suffix: str = ".png"
    ) -> Path:
        """
        Save individual page image in strict sequential order:
        storage/processed/<doc_name>/page_001.png, page_002.png...
        Padded with leading zeros to maintain exact alphabetical and numerical sort order.
        """
        clean_stem = Path(document_name).stem
        doc_folder = self.processed_dir / clean_stem
        doc_folder.mkdir(parents=True, exist_ok=True)

        dest_path = doc_folder / f"page_{page_num:03d}{suffix}"

        if isinstance(image_data, np.ndarray):
            pil_img = Image.fromarray(image_data.astype(np.uint8))
        else:
            pil_img = image_data

        pil_img.save(dest_path)
        return dest_path

    def save_processed_pdf(self, pdf_bytes_or_doc: Any, document_name: str) -> Path:
        """Save enhanced/re-assembled PDF to storage/processed/enhanced_<doc_name>.pdf."""
        clean_stem = Path(document_name).stem
        dest_path = self.processed_dir / f"enhanced_{clean_stem}.pdf"

        if isinstance(pdf_bytes_or_doc, bytes):
            dest_path.write_bytes(pdf_bytes_or_doc)
        elif hasattr(pdf_bytes_or_doc, "save"):
            # fitz.Document
            pdf_bytes_or_doc.save(str(dest_path))
        else:
            raise TypeError("pdf_bytes_or_doc must be bytes or a fitz.Document object")

        return dest_path


    def save_report(self, assessment_data: Union[Dict[str, Any], Any], identifier: str) -> Path:
        """Save quality assessment report JSON to storage/reports/."""
        if hasattr(assessment_data, "model_dump"):
            data = assessment_data.model_dump()
        elif isinstance(assessment_data, dict):
            data = assessment_data
        else:
            raise TypeError("assessment_data must be dict or Pydantic model")

        dest_path = self.reports_dir / f"{identifier}.json"
        dest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return dest_path

    def get_report(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Retrieve previously saved report JSON by identifier or hash."""
        report_file = self.reports_dir / f"{identifier}.json"
        if not report_file.exists():
            return None
        try:
            return json.loads(report_file.read_text(encoding="utf-8"))
        except Exception:
            return None
