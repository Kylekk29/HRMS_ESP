from embedding_mgr import EmbeddingManager
from model_provider import AIModelProvider
import os
import config

class MainLogicHandler:
    def __init__(self):
        self.emb_mgr = EmbeddingManager()
        self.ai = AIModelProvider()

    def handle_request(self, request_data):
        task_type = request_data.get("task")
        candidate_id = request_data.get("candidate_id")
        jd_text = request_data.get("jd")
        file_path = request_data.get("file_path")

        if task_type == "cv_screening":
            db_path = os.path.join(config.DB_ROOT, candidate_id)
            
            # 【Debug 優化】：若資料庫已存在，則跳過 Embedding 步驟以節省時間
            if not os.path.exists(db_path):
                print(f"Creating new vector DB for {candidate_id}...")
                self.emb_mgr.embed_cv(file_path, candidate_id)
            
            # 2. 檢索與 JD 最相關的履歷內容
            db = self.emb_mgr.load_db(candidate_id)
            docs = db.similarity_search(jd_text, k=4)
            context = "\n".join([d.page_content for d in docs])
            print(f"Retrieved context for {candidate_id}:\n{context}\n")
            
            # 3. 呼叫 AI 進行推理評分
            return self.ai.ask_ai(context, jd_text)
        
        return "Feature not implemented"