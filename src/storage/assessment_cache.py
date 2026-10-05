"""
Assessment Cache and Deduplication Engine for AI Document Quality Gate.
Uses SQLite and SHA-256 content hashing to provide instantaneous (<1ms) idempotent results
when identical documents or contracts are submitted repeatedly.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import hashlib
import sqlite3
import json
from datetime import datetime


class AssessmentCache:
    """Persistent SQLite cache storing evaluation results keyed by SHA-256 file content hash."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        if db_path is None:
            # Default to <project_root>/storage/cache/assessments.db
            storage_dir = Path(__file__).resolve().parent.parent.parent / "storage" / "cache"
            storage_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = storage_dir / "assessments.db"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create assessment_cache table if not exists."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS assessment_cache (
                    file_hash TEXT PRIMARY KEY,
                    file_name TEXT,
                    file_size INTEGER,
                    assessment_json TEXT NOT NULL,
                    processed_file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_hash ON assessment_cache(file_hash);")
            conn.commit()

    @staticmethod
    def compute_file_hash(file_path_or_bytes: Union[str, Path, bytes], chunk_size: int = 65536) -> str:
        """Compute SHA-256 hash using chunked streaming for memory efficiency."""
        sha256 = hashlib.sha256()
        if isinstance(file_path_or_bytes, bytes):
            sha256.update(file_path_or_bytes)
        else:
            path = Path(file_path_or_bytes)
            with open(path, "rb") as f:
                while chunk := f.read(chunk_size):
                    sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def compute_cache_key(file_hash: str, config_hash: Optional[str] = None) -> str:
        """Combine file content hash with configuration hash for automatic invalidation upon threshold changes."""
        if not config_hash:
            return file_hash
        return hashlib.sha256(f"{file_hash}:{config_hash}".encode("utf-8")).hexdigest()

    def get(self, file_hash: str, config_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached assessment data by file hash and optional config hash."""
        cache_key = self.compute_cache_key(file_hash, config_hash)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT assessment_json, processed_file_path, updated_at FROM assessment_cache WHERE file_hash = ?",
                (cache_key,)
            )

            row = cursor.fetchone()
            if row is None:
                return None

            try:
                data = json.loads(row["assessment_json"])
                data["_cached"] = True
                data["_cached_at"] = row["updated_at"]
                if row["processed_file_path"]:
                    data["_cached_processed_path"] = row["processed_file_path"]
                return data
            except Exception:
                return None

    def set(
        self,
        file_hash: str,
        file_name: str,
        file_size: int,
        assessment_data: Union[Dict[str, Any], Any],
        processed_file_path: Optional[str] = None,
        config_hash: Optional[str] = None
    ) -> None:
        """Save or update assessment result in cache."""
        cache_key = self.compute_cache_key(file_hash, config_hash)
        if hasattr(assessment_data, "model_dump"):
            json_text = json.dumps(assessment_data.model_dump(), ensure_ascii=False)
        elif isinstance(assessment_data, dict):
            json_text = json.dumps(assessment_data, ensure_ascii=False)
        else:
            json_text = str(assessment_data)

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO assessment_cache (file_hash, file_name, file_size, assessment_json, processed_file_path, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_name = excluded.file_name,
                    file_size = excluded.file_size,
                    assessment_json = excluded.assessment_json,
                    processed_file_path = excluded.processed_file_path,
                    updated_at = CURRENT_TIMESTAMP;
            """, (cache_key, file_name, file_size, json_text, processed_file_path))
            conn.commit()


    def delete(self, file_hash: str) -> bool:
        """Remove a record from cache."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM assessment_cache WHERE file_hash = ?", (file_hash,))
            conn.commit()
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Clear entire assessment cache."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM assessment_cache;")
            conn.commit()
