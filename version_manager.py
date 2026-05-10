import hashlib
import json
import os
import shutil
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import config

logger = logging.getLogger(__name__)


class VersionManager:
    """
    Tracks file content versions using SHA-256 hashes.
    - Same content → reuse existing vector DB (skip re-embedding)
    - Different content (same name) → create new DB ID, archive old file
    - Always points to latest version

    FIX #1: check_and_register_version is now split into two phases:
      1. check_version()        — read-only, no disk writes
      2. commit_new_version()   — write metadata ONLY after FAISS index is saved

    FIX #2: check_version() validates that the FAISS directory actually exists on
    disk before returning is_new=False (reuse). If the index is missing it signals
    the caller to re-embed.
    """

    def __init__(self):
        logger.info("Initializing VersionManager...")
        self.metadata_dir = config.METADATA_DIR
        self.archive_dir = config.ARCHIVE_DIR
        logger.info(f"  Metadata dir: {self.metadata_dir}")
        logger.info(f"  Archive dir: {self.archive_dir}")

    # ──────────────────────────────────────────────
    # Hashing
    # ──────────────────────────────────────────────

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file."""
        logger.info(f"Calculating hash for: {os.path.basename(file_path)}")
        start_time = time.time()
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        file_hash = hasher.hexdigest()
        elapsed = time.time() - start_time
        logger.info(f"  Hash: {file_hash[:16]}... calculated in {elapsed:.3f}s")
        return file_hash

    # ──────────────────────────────────────────────
    # Metadata paths
    # ──────────────────────────────────────────────

    def _meta_path(self, file_identifier: str) -> str:
        safe_id = file_identifier.replace("/", "_").replace("\\", "_")
        return os.path.join(self.metadata_dir, f"{safe_id}_meta.json")

    def _archive_path(self, file_identifier: str, file_hash: str, ext: str) -> str:
        safe_id = file_identifier.replace("/", "_").replace("\\", "_")
        return os.path.join(self.archive_dir, f"{safe_id}_{file_hash[:16]}{ext}")

    # ──────────────────────────────────────────────
    # Load / Save
    # ──────────────────────────────────────────────

    def load_metadata(self, file_identifier: str) -> Dict:
        path = self._meta_path(file_identifier)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.info(
                f"Loaded metadata for {file_identifier}: "
                f"{len(metadata.get('versions', []))} versions"
            )
            return metadata
        logger.info(f"No existing metadata for {file_identifier}")
        return {
            "file_id": file_identifier,
            "versions": [],
            "current_hash": None,
            "current_db_id": None,
        }

    def save_metadata(self, file_identifier: str, metadata: Dict) -> None:
        path = self._meta_path(file_identifier)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata for {file_identifier}")

    # ──────────────────────────────────────────────
    # Phase 1: Read-only version check
    # ──────────────────────────────────────────────

    def check_version(
        self, file_path: str, file_identifier: str, category: str,
        base_dir: Optional[str] = None,
    ) -> Dict:
        """
        READ-ONLY check — no metadata is written here.

        Returns a dict with:
          is_new        : bool   — True if a new index must be built
          file_hash     : str
          db_id         : str    — existing if reused, proposed if new
          action        : "reused" | "created" | "updated" | "repaired"
          version_number: int
          message       : str
          _internal     : dict   — private fields consumed by commit_new_version

        FIX #2: When the hash matches an existing version, the method checks whether
        the corresponding FAISS directory actually exists on disk.  If it is missing,
        action is set to "repaired" and is_new=True so the caller rebuilds the index
        without creating a duplicate metadata entry.
        """
        logger.info(f"Checking version for: {file_identifier}")
        logger.info(f"  Category: {category}")

        start_time = time.time()
        file_hash = self.calculate_file_hash(file_path)
        metadata = self.load_metadata(file_identifier)

        # ── Duplicate content check ──
        for version in metadata["versions"]:
            if version["hash"] == file_hash:
                db_id = version["db_id"]
                logger.info(
                    f"  ✅ DUPLICATE FOUND (v{version['version']}, "
                    f"from {version['timestamp'][:10]})"
                )
                logger.info(f"  Reusing DB ID: {db_id}")

                # FIX #2 – validate the index actually lives on disk
                if base_dir is not None:
                    index_file = os.path.join(base_dir, db_id, "index.faiss")
                    if not os.path.exists(index_file):
                        logger.warning(
                            f"  ⚠️  FAISS index missing on disk for db_id={db_id} "
                            f"— forcing rebuild (action=repaired)"
                        )
                        # Treat as new build but do NOT create a new version entry;
                        # the existing metadata entry is correct — we just need the
                        # index file to be regenerated.
                        _, ext = os.path.splitext(file_path)
                        return {
                            "is_new": True,
                            "file_hash": file_hash,
                            "db_id": db_id,
                            "action": "repaired",
                            "version_number": version["version"],
                            "message": (
                                f"Index missing on disk — rebuilding "
                                f"(v{version['version']}, db_id={db_id})"
                            ),
                            "_internal": {
                                "skip_commit": True,  # metadata already exists
                                "file_path": file_path,
                                "archive_path": self._archive_path(
                                    file_identifier, file_hash, ext
                                ),
                                "category": category,
                                "is_update": True,
                            },
                        }

                return {
                    "is_new": False,
                    "file_hash": file_hash,
                    "db_id": db_id,
                    "action": "reused",
                    "version_number": version["version"],
                    "message": (
                        f"Identical content detected — reusing existing index "
                        f"(v{version['version']}, created {version['timestamp'][:10]})"
                    ),
                    "_internal": None,
                }

        # ── New version — propose IDs without writing anything ──
        logger.info("  🆕 NEW VERSION DETECTED")
        _, ext = os.path.splitext(file_path)
        version_number = len(metadata["versions"]) + 1
        db_id = f"{file_identifier}_{file_hash[:12]}"
        is_update = len(metadata["versions"]) > 0
        action = "updated" if is_update else "created"
        archive_path = self._archive_path(file_identifier, file_hash, ext)

        logger.info(f"  New version: v{version_number}")
        logger.info(f"  DB ID: {db_id}")

        msg = (
            f"New version registered (v{version_number}, DB: {db_id})"
            if not is_update
            else f"Updated to v{version_number} — previous version archived"
        )

        elapsed = time.time() - start_time
        logger.info(f"  Version check completed in {elapsed:.2f}s")

        return {
            "is_new": True,
            "file_hash": file_hash,
            "db_id": db_id,
            "action": action,
            "version_number": version_number,
            "message": msg,
            "_internal": {
                "skip_commit": False,
                "file_path": file_path,
                "archive_path": archive_path,
                "category": category,
                "is_update": is_update,
            },
        }

    # ──────────────────────────────────────────────
    # Phase 2: Commit metadata AFTER successful index save
    # ──────────────────────────────────────────────

    def commit_new_version(self, file_identifier: str, check_result: Dict) -> None:
        """
        Persist version metadata to disk.  Must be called ONLY after the FAISS
        index has been successfully saved.

        If check_result["_internal"]["skip_commit"] is True (repaired case) this
        is a no-op because the metadata entry already exists.
        """
        internal = check_result.get("_internal") or {}
        if internal.get("skip_commit"):
            logger.info(
                f"  ℹ️  commit_new_version: skipping (repaired case, "
                f"metadata already exists for {file_identifier})"
            )
            return

        file_path = internal["file_path"]
        archive_path = internal["archive_path"]
        category = internal["category"]

        # Archive the source file
        if not os.path.exists(archive_path):
            logger.info(f"  Archiving to: {archive_path}")
            try:
                shutil.copy2(file_path, archive_path)
            except Exception as e:
                logger.error(f"  ❌ Failed to archive file: {e}")
                raise RuntimeError(f"Failed to archive file {file_path}: {e}")

        # Re-load metadata (could have been modified by a concurrent call)
        metadata = self.load_metadata(file_identifier)

        # Guard: don't add a duplicate entry if another process already committed
        for v in metadata["versions"]:
            if v["hash"] == check_result["file_hash"]:
                logger.info(
                    f"  ℹ️  commit_new_version: version already present "
                    f"(concurrent commit?), skipping."
                )
                return

        version_entry = {
            "version": check_result["version_number"],
            "hash": check_result["file_hash"],
            "db_id": check_result["db_id"],
            "timestamp": datetime.now().isoformat(),
            "archive_path": archive_path,
            "category": category,
        }

        metadata["versions"].append(version_entry)
        metadata["current_hash"] = check_result["file_hash"]
        metadata["current_db_id"] = check_result["db_id"]

        # Prune old versions
        while len(metadata["versions"]) > config.MAX_VERSIONS_PER_FILE:
            old = metadata["versions"].pop(0)
            old_archive = old.get("archive_path", "")
            logger.info(f"  Pruning old version: v{old['version']} ({old['hash'][:12]}...)")
            if old_archive and os.path.exists(old_archive):
                try:
                    os.remove(old_archive)
                    logger.info(f"  ✓ Removed old archive: {old_archive}")
                except OSError as e:
                    logger.warning(f"  ⚠️ Could not remove old archive {old_archive}: {e}")
            else:
                logger.info(f"  Archive file not found, skipping deletion: {old_archive}")

        try:
            self.save_metadata(file_identifier, metadata)
        except Exception as e:
            logger.error(f"  ❌ Failed to save metadata: {e}")
            raise RuntimeError(f"Failed to save version metadata: {e}")

        logger.info(
            f"  ✅ Version committed: v{check_result['version_number']} "
            f"for {file_identifier}"
        )

    # ──────────────────────────────────────────────
    # Legacy wrapper (kept for backward compatibility)
    # ──────────────────────────────────────────────

    def check_and_register_version(
        self, file_path: str, file_identifier: str, category: str
    ) -> Dict:
        """
        Deprecated: use check_version() + commit_new_version() instead.
        This wrapper preserves the old single-call API but DOES NOT fix the
        atomicity bug — callers should migrate to the two-phase API.
        """
        logger.warning(
            "check_and_register_version() is deprecated; "
            "use check_version() + commit_new_version() for atomicity."
        )
        result = self.check_version(file_path, file_identifier, category)
        if result["is_new"] and not (result.get("_internal") or {}).get("skip_commit"):
            self.commit_new_version(file_identifier, result)
        return result

    # ──────────────────────────────────────────────
    # History
    # ──────────────────────────────────────────────

    def get_version_history(self, file_identifier: str) -> List[Dict]:
        logger.info(f"Retrieving version history for: {file_identifier}")
        history = self.load_metadata(file_identifier)["versions"]
        logger.info(f"  Found {len(history)} versions")
        return history

    def get_current_db_id(self, file_identifier: str) -> Optional[str]:
        db_id = self.load_metadata(file_identifier).get("current_db_id")
        if db_id:
            logger.info(f"Current DB ID for {file_identifier}: {db_id}")
        else:
            logger.warning(f"No current DB ID for {file_identifier}")
        return db_id