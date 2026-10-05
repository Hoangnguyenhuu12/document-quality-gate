"""
FastAPI entrypoint for AI Document Quality Gate.
Provides HTTP REST endpoints for OCR readiness assessment.
"""

from pathlib import Path
import shutil
import tempfile
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.quality_pipeline import DocumentQualityPipeline
from src.schemas.quality_assessment import DocumentQualityAssessment

app = FastAPI(
    title="Document Quality Gate API",
    version="1.1.0",
    description="Document Quality Gate service inspecting blur, contrast, resolution, shadow, skew tilt, and glare."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = DocumentQualityPipeline()


class AssessPathRequest(BaseModel):
    file_path: str
    max_pages: int = 10
    use_cache: bool = True
    save_report: bool = True


@app.get("/health")
def health():
    return {"status": "healthy", "service": "document-quality-gate", "version": "1.2.0"}


@app.get("/storage/report/{identifier}")
def get_stored_report(identifier: str):
    """Retrieve persistent JSON report by file hash or identifier."""
    report = pipeline.storage.get_report(identifier)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report not found for identifier: {identifier}")
    return report


@app.post("/quality/assess", response_model=DocumentQualityAssessment)
async def assess_quality(
    file: UploadFile = File(None),
    file_path: str = Form(None),
    max_pages: int = Form(10),
    use_cache: bool = Form(True),
    save_report: bool = Form(True),
    store_upload: bool = Form(True)
):
    """
    Assess document quality and return OCR readiness decision.
    Accepts uploaded file or local file_path.
    Utilizes SHA-256 deduplication cache and persists to storage/ directories.
    """
    temp_path = None
    try:
        if file is not None:
            filename = file.filename or "uploaded_doc.pdf"
            suffix = Path(filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = Path(tmp.name)

            if store_upload:
                stored_file = pipeline.storage.save_upload(temp_path, filename)
                target_path = stored_file
            else:
                target_path = temp_path
        elif file_path:
            target_path = Path(file_path)
            if not target_path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        else:
            raise HTTPException(status_code=400, detail="Must provide either 'file' or 'file_path'")

        result = pipeline.assess_document(
            target_path,
            max_pages=max_pages,
            use_cache=use_cache,
            save_report=save_report
        )
        return result
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.post("/quality/assess-path", response_model=DocumentQualityAssessment)
def assess_by_path(req: AssessPathRequest):
    """Assess document quality via JSON body specifying local file path."""
    target_path = Path(req.file_path)
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")
    return pipeline.assess_document(
        target_path,
        max_pages=req.max_pages,
        use_cache=req.use_cache,
        save_report=req.save_report
    )


@app.post("/quality/enhance")
async def enhance_image(
    file: UploadFile = File(...)
):
    """
    Auto-enhances a degraded/skewed/shadowed document image (deskew, shadow removal, contrast stretch).
    Saves a persistent copy into storage/processed/ and returns the cleaned image file.
    """
    suffix = Path(file.filename or "image.png").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        shutil.copyfileobj(file.file, tmp_in)
        temp_in_path = Path(tmp_in.name)

    try:
        # Save to storage/processed
        enhanced_path = pipeline.enhance_image_file(temp_in_path, output_path=None)
        return FileResponse(
            enhanced_path,
            media_type="image/png",
            filename=f"enhanced_{file.filename or 'doc.png'}"
        )
    finally:
        if temp_in_path.exists():
            temp_in_path.unlink(missing_ok=True)


