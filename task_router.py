import os
import config
from model_provider import AIModelProvider
from embedding_mgr import EmbeddingManager

class TaskRouter:
    def __init__(self):
        self.ai = AIModelProvider()
        self.emb = EmbeddingManager()

    async def batch_screen(self, jd: str, safe_ids: list):
        combined_contexts = []
        
        for cid in safe_ids:
            try:
                db = self.emb.load_db(cid)
                # k=3 is usually enough for resume context to stay under token limits
                docs = db.similarity_search(jd, k=3) 
                text = "\n".join([d.page_content for d in docs])
                combined_contexts.append(f"--- 候選人 ID: {cid} ---\n{text}")
            except Exception as e:
                print(f"Skipping {cid} due to error: {e}")
        
        full_ctx = "\n\n".join(combined_contexts)
        return self.ai.ask_ai(full_ctx, "cv_screening", jd=jd)

    async def execute(self, task_type: str, data: dict):
        cid = data.get("candidate_id", "")
        try:
            db = self.emb.load_db(cid)
            query = data.get("query", "人才評估")
            docs = db.similarity_search(query, k=5)
            context = "\n".join([d.page_content for d in docs])
            return self.ai.ask_ai(context, task_type, **data)
        except Exception as e:
            return {"error": f"Database error: {str(e)}"}