import hashlib
import logging
import os
import re
import time
from typing import Dict, List, Optional
from functools import wraps
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

import config
from version_manager import VersionManager

logger = logging.getLogger(__name__)


def timed(func):
    """Decorator to log execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"⏱️  Starting {func.__name__}...")
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"✅ Completed {func.__name__} in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Failed {func.__name__} after {elapsed:.2f}s: {str(e)}")
            raise
    return wrapper


class EmbeddingManager:
    """
    Handles document loading, chunking, embedding, and FAISS persistence.

    DB layout:
      ./vector_dbs/
        cv_candidates/<db_id>/
        company_culture/<db_id>/
        employees/<employee_id>/cv/<db_id>/       ← CV for this employee
        employees/<employee_id>/profile/<db_id>/   ← Other profile docs

    Key fixes applied
    ─────────────────
    FIX #1  embed_file_with_versioning now follows the order:
              check_version (read-only) → build FAISS → save FAISS → commit_new_version
            Metadata is written ONLY after the index is successfully persisted to disk.
            On embedding failure the version entry is never created, so re-uploading
            the same file will correctly trigger a fresh embed.

    FIX #2  On a "reused" version hit, the db_path is verified on disk before
            returning. If the index is gone, the file is transparently re-embedded
            (the existing metadata entry is preserved — no duplicate version).
            This is handled inside VersionManager.check_version; embedding_mgr
            simply acts on the is_new=True / action="repaired" signal.

    FIX #3  clean_id() appends a short content-hash suffix whenever sanitisation
            changes the original string, preventing different raw IDs from mapping
            to the same filesystem path (e.g. "user@123" vs "user_123").
    """

    def __init__(self):
        logger.info("=" * 60)
        logger.info("Initializing EmbeddingManager...")

        start_time = time.time()
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        logger.info(f"Device: {config.EMBEDDING_DEVICE}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"Embedding model loaded in {time.time() - start_time:.2f}s")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
        )
        logger.info(
            f"Text splitter configured: chunk_size={config.CHUNK_SIZE}, "
            f"overlap={config.CHUNK_OVERLAP}"
        )

        self.version_mgr = VersionManager()
        logger.info("EmbeddingManager initialized successfully")
        logger.info("=" * 60)

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def clean_id(raw: str) -> str:
        """
        Sanitise a string for use as a filesystem-safe identifier.
        Keeps alphanumeric + underscore + hyphen; strips file extension.

        FIX #3: If sanitisation changes the original string (i.e. special
        characters were replaced), a 6-char MD5 suffix is appended so that
        two different raw IDs that collapse to the same cleaned string remain
        distinguishable.
        Examples:
          "EMP-001"   → "EMP-001"          (no change → no suffix)
          "EMP_001"   → "EMP_001"          (no change → no suffix)
          "user@123"  → "user_123_a3f9c1"  (@ replaced → suffix added)
          "user_123"  → "user_123"         (no change → no suffix)
        """
        name = os.path.splitext(os.path.basename(raw))[0]
        cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")

        if not cleaned:
            return f"file_{abs(hash(raw)) % 100000}"

        # FIX #3: add disambiguation suffix only when the string was changed
        if cleaned != name:
            short_hash = hashlib.md5(raw.encode("utf-8", errors="replace")).hexdigest()[:6]
            cleaned = f"{cleaned}_{short_hash}"

        return cleaned

    # Keep legacy alias
    def clean_file_name(self, file_name: str) -> str:
        return self.clean_id(file_name)

    @timed
    def _load_document(self, file_path: str):
        """Load document based on file extension."""
        ext = os.path.splitext(file_path)[-1].lower()
        file_size = os.path.getsize(file_path)
        logger.info(f"Loading document: {file_path} ({ext}, {file_size / 1024:.1f} KB)")

        loaders = {
            ".pdf": lambda p: PyPDFLoader(p),
            ".txt": lambda p: TextLoader(p, encoding="utf-8"),
            ".docx": lambda p: Docx2txtLoader(p),
        }
        if ext not in loaders:
            raise ValueError(
                f"Unsupported file type: {ext}  (supported: pdf, txt, docx)"
            )

        docs = loaders[ext](file_path).load()
        logger.info(f"Document loaded: {len(docs)} pages/sections")
        return docs

    # ──────────────────────────────────────────────
    # Core: embed with versioning
    # ──────────────────────────────────────────────

    @timed
    def embed_file_with_versioning(
        self,
        file_path: str,
        file_identifier: str,
        base_dir: str,
        category: str,
    ) -> Dict:
        """
        Embed a file into a FAISS index under base_dir/<db_id>.
        Skips re-embedding if the file content hasn't changed AND the index
        file is present on disk.

        Operation order (FIX #1):
          1. check_version()   — read-only hash check, no disk writes
          2. If reused + index exists on disk → return immediately
          3. Build FAISS index in memory
          4. Save index to disk
          5. commit_new_version() — write metadata ONLY after step 4 succeeds
          6. On any failure between steps 3-4 → metadata is NOT written;
             re-uploading the same file will trigger a clean embed.

        Returns:
          {
            "db_id"         : str,
            "db_path"       : str,
            "is_new"        : bool,
            "action"        : "created" | "reused" | "updated" | "repaired",
            "message"       : str,
            "version_number": int,
            "chunk_count"   : int   (only when newly embedded)
          }
        """
        logger.info(f"Embedding file: {os.path.basename(file_path)}")
        logger.info(f"  Identifier: {file_identifier}")
        logger.info(f"  Category:   {category}")
        logger.info(f"  Base dir:   {base_dir}")

        # ── Step 1: Read-only version check (FIX #1 & FIX #2) ──
        start_time = time.time()
        version_info = self.version_mgr.check_version(
            file_path, file_identifier, category, base_dir=base_dir
        )
        db_id = version_info["db_id"]
        db_path = os.path.join(base_dir, db_id)

        logger.info(f"Version check completed in {time.time() - start_time:.2f}s")
        logger.info(f"  DB ID:   {db_id}")
        logger.info(f"  Is new:  {version_info['is_new']}")
        logger.info(f"  Action:  {version_info['action']}")

        # ── Step 2: Reuse — index confirmed present on disk ──
        if not version_info["is_new"]:
            logger.info(f"✅ Reusing existing index at: {db_path}")
            return {**version_info, "db_path": db_path}

        # ── Steps 3-4: Build and save FAISS index BEFORE touching metadata ──
        logger.info(
            f"🔄 Building FAISS index "
            f"({'repair' if version_info['action'] == 'repaired' else 'new'})..."
        )
        try:
            docs = self._load_document(file_path)
            logger.info("Splitting into chunks...")
            split_docs = self.text_splitter.split_documents(docs)

            if not split_docs:
                raise ValueError("Document is empty or could not be parsed.")

            logger.info(f"Created {len(split_docs)} chunks")
            logger.info(f"Generating embeddings for {len(split_docs)} chunks...")
            chunk_start = time.time()

            db = FAISS.from_documents(split_docs, self.embeddings)

            logger.info(f"Embeddings generated in {time.time() - chunk_start:.2f}s")
            logger.info(f"Saving FAISS index to: {db_path}")

            os.makedirs(db_path, exist_ok=True)
            db.save_local(db_path)

            index_size = os.path.getsize(os.path.join(db_path, "index.faiss"))
            logger.info(f"✅ Index saved ({index_size / 1024 / 1024:.1f} MB)")

        except Exception as exc:
            # FIX #1: metadata is NOT written — the caller can retry cleanly
            logger.error(
                f"❌ Embedding failed — metadata NOT updated. "
                f"Re-uploading the same file will trigger a fresh embed. "
                f"Error: {exc}",
                exc_info=True,
            )
            return {
                **version_info,
                "db_path": db_path,
                "error": str(exc),
            }

        # ── Step 5: Commit version metadata (FIX #1) ──
        # Skip commit for "repaired" case — the entry already exists in metadata.
        internal = version_info.get("_internal") or {}
        if not internal.get("skip_commit"):
            try:
                self.version_mgr.commit_new_version(file_identifier, version_info)
            except Exception as exc:
                # Index is on disk but metadata write failed.
                # Log prominently; the index is usable but version tracking is
                # inconsistent — operator should investigate.
                logger.error(
                    f"❌ FAISS index saved but metadata commit failed for "
                    f"{file_identifier}: {exc}. "
                    f"The index at {db_path} is usable but version history may "
                    f"be incomplete.",
                    exc_info=True,
                )
                # Return partial success so the caller can still use the index
                return {
                    **version_info,
                    "db_path": db_path,
                    "chunk_count": len(split_docs),
                    "warning": f"Index saved but metadata commit failed: {exc}",
                }

        return {
            **version_info,
            "db_path": db_path,
            "chunk_count": len(split_docs),
        }

    # ──────────────────────────────────────────────
    # Structured employee folder embedding
    # ──────────────────────────────────────────────

    @timed
    def embed_employee_document(
        self,
        file_path: str,
        employee_id: str,
        doc_type: str = "profile",
    ) -> Dict:
        """
        Embed a document specifically for an employee.
        doc_type: "cv" | "profile" | "review" | "other"

        DB path: ./vector_dbs/employees/<employee_id>/<doc_type>/<db_id>/
        """
        logger.info("Embedding employee document:")
        logger.info(f"  Employee ID:   {employee_id}")
        logger.info(f"  Document type: {doc_type}")

        safe_emp_id = self.clean_id(employee_id)
        employee_base = os.path.join(config.EMPLOYEE_DB_DIR, safe_emp_id, doc_type)
        os.makedirs(employee_base, exist_ok=True)

        file_id = f"{safe_emp_id}_{doc_type}"
        logger.info(f"  Employee base: {employee_base}")

        return self.embed_file_with_versioning(
            file_path, file_id, employee_base, f"employee_{doc_type}"
        )

    # ──────────────────────────────────────────────
    # Batch CV embedding (up to MAX_CV_BATCH_SIZE)
    # ──────────────────────────────────────────────

    @timed
    def embed_cv_batch(self, cv_files: List[Dict]) -> List[Dict]:
        """
        Embed multiple CV files.  Each item: {"file_path": str, "file_name": str}
        Returns list of results in the same order.
        Caps at config.MAX_CV_BATCH_SIZE.
        """
        logger.info(f"Starting batch CV embedding: {len(cv_files)} files")
        logger.info(f"Max batch size: {config.MAX_CV_BATCH_SIZE}")

        if len(cv_files) > config.MAX_CV_BATCH_SIZE:
            logger.warning(
                f"CV batch size ({len(cv_files)}) exceeds max limit "
                f"({config.MAX_CV_BATCH_SIZE}). "
                f"Only the first {config.MAX_CV_BATCH_SIZE} will be processed."
            )

        results = []
        cv_files_limited = cv_files[: config.MAX_CV_BATCH_SIZE]

        for idx, cv in enumerate(cv_files_limited, 1):
            logger.info(
                f"Processing CV {idx}/{len(cv_files_limited)}: {cv['file_name']}"
            )
            file_id = self.clean_id(cv["file_name"])

            file_start = time.time()
            result = self.embed_file_with_versioning(
                cv["file_path"], file_id, config.CV_DB_DIR, "cv"
            )
            file_elapsed = time.time() - file_start

            result["file_name"] = cv["file_name"]
            result["file_id"] = file_id
            results.append(result)

            logger.info(
                f"Processed CV {idx}/{len(cv_files_limited)}: "
                f"{cv['file_name']} in {file_elapsed:.2f}s"
            )
            logger.info(f"  → DB ID:  {result.get('db_id', 'N/A')}")
            logger.info(f"  → Action: {result.get('action', 'N/A')}")
            if "chunk_count" in result:
                logger.info(f"  → Chunks: {result['chunk_count']}")

        logger.info(f"Batch CV embedding complete: {len(results)} processed")
        return results

    # ──────────────────────────────────────────────
    # Load / search helpers
    # ──────────────────────────────────────────────

    @timed
    def load_db(self, db_id: str, base_dir: str) -> FAISS:
        """Load a FAISS index from disk."""
        db_path = os.path.join(base_dir, db_id)
        logger.info(f"Loading FAISS index: {db_id}")
        logger.info(f"  Path: {db_path}")

        index_file = os.path.join(db_path, "index.faiss")
        if not os.path.exists(index_file):
            logger.error(f"FAISS index not found at: {db_path}")
            raise FileNotFoundError(f"FAISS index not found at: {db_path}")

        index_size = os.path.getsize(index_file)
        logger.info(f"Index file size: {index_size / 1024 / 1024:.1f} MB")

        db = FAISS.load_local(
            db_path, self.embeddings, allow_dangerous_deserialization=True
        )
        logger.info("✅ Index loaded successfully")
        return db

    @timed
    def load_employee_db(self, employee_id: str, doc_type: str = "profile") -> FAISS:
        """Load the latest embedded DB for an employee document type."""
        logger.info(f"Loading employee DB: {employee_id} ({doc_type})")

        safe_emp_id = self.clean_id(employee_id)
        employee_base = os.path.join(config.EMPLOYEE_DB_DIR, safe_emp_id, doc_type)
        file_id = f"{safe_emp_id}_{doc_type}"

        db_id = self.version_mgr.get_current_db_id(file_id)
        if not db_id:
            logger.error(
                f"No embedded {doc_type} found for employee: {employee_id}"
            )
            raise FileNotFoundError(
                f"No embedded {doc_type} found for employee: {employee_id}"
            )

        # FIX #2 — validate the index file exists before attempting load
        index_file = os.path.join(employee_base, db_id, "index.faiss")
        if not os.path.exists(index_file):
            logger.error(
                f"Metadata for {employee_id} ({doc_type}) points to db_id={db_id} "
                f"but the index file is missing at {index_file}. "
                f"The employee document must be re-uploaded to rebuild the index."
            )
            raise FileNotFoundError(
                f"FAISS index for employee {employee_id} ({doc_type}) is missing on "
                f"disk (db_id={db_id}). Please re-upload the document."
            )

        logger.info(f"Found DB ID: {db_id}")
        return self.load_db(db_id, employee_base)

    def get_version_history(self, file_identifier: str) -> list:
        logger.info(f"Retrieving version history for: {file_identifier}")
        history = self.version_mgr.get_version_history(file_identifier)
        logger.info(f"Found {len(history)} versions")
        return history
