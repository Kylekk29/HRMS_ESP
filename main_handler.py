"""
main_handler.py
───────────────
Legacy single-request handler (kept for backward compatibility).
For production use, prefer TaskRouter via main_api.py.
"""

from embedding_mgr import EmbeddingManager
from model_provider import AIModelProvider
import os
import config


class MainLogicHandler:
    def __init__(self):
        self.emb_mgr = EmbeddingManager()
        self.ai = AIModelProvider()

    def handle_request(self, request_data: dict):
        task_type    = request_data.get("task")
        candidate_id = request_data.get("candidate_id", "")
        jd_text      = request_data.get("jd", "")
        file_path    = request_data.get("file_path", "")

        if task_type == "cv_screening":
            if not candidate_id or not jd_text:
                return {"error": "candidate_id and jd are required for cv_screening"}

            # Embed with version control
            result = self.emb_mgr.embed_file_with_versioning(
                file_path,
                self.emb_mgr.clean_id(candidate_id),
                config.CV_DB_DIR,
                "cv",
            )

            if "error" in result:
                return {"error": result["error"]}

            db = self.emb_mgr.load_db(result["db_id"], config.CV_DB_DIR)
            docs = db.similarity_search(jd_text, k=config.CV_RETRIEVAL_K)
            context = "\n".join(d.page_content for d in docs)

            return self.ai.cv_screening_ai(
                context,
                "cv_screening",
                jd=jd_text,
                culture_ctx="Not provided",
            )

        return {"error": f"Unknown task type: {task_type}"}
