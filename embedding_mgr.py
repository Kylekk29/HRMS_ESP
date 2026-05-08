import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import config

class EmbeddingManager:
    def __init__(self):
        # Fallback to a public model if local path doesn't exist
        model_name = config.EMBEDDING_MODEL_PATH
        if not os.path.exists(model_name):
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", " "]
        )

    def clean_candidate_id(self, file_name: str) -> str:
        """
        解決 Windows + FAISS 的中文路徑 Bug：
        將 ID 轉換為純英數雜湊，避免底層 C++ 庫寫入失敗。
        """
        # 取得純檔名 (不含副檔名)
        name_only = os.path.splitext(os.path.basename(file_name))[0]
        
        # 產生 MD5 雜湊以確保唯一性與純英數路徑
        # 使用 utf-8 編碼字串後進行雜湊
        hash_digest = hashlib.md5(name_only.encode('utf-8')).hexdigest()
        
        # 為了除錯方便，我們可以保留前幾個字母(僅限英數)，後面接雜湊
        readable = re.sub(r'[^\x00-\x7F]+', '', name_only) # 移除所有非 ASCII (中文)
        readable = re.sub(r'[^a-zA-Z0-9]', '', readable)[:10] # 只留英數
        
        if not readable:
            return hash_digest[:16]
        return f"{readable}_{hash_digest[:8]}"

    def load_document(self, file_path: str):
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file: {ext}")
        return loader.load()

    def embed_cv(self, file_path: str, candidate_id: str) -> str:
        db_path = os.path.join(config.DB_ROOT, candidate_id)
        
        # Skip if already exists
        if os.path.exists(os.path.join(db_path, "index.faiss")):
            return candidate_id

        docs = self.load_document(file_path)
        split_docs = self.text_splitter.split_documents(docs)
        
        db = FAISS.from_documents(split_docs, self.embeddings)
        os.makedirs(db_path, exist_ok=True) 
        db.save_local(db_path)
        return candidate_id

    def load_db(self, candidate_id: str):
        # We assume candidate_id is already cleaned here
        db_path = os.path.join(config.DB_ROOT, candidate_id)
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Vector DB not found for {candidate_id}")
        
        return FAISS.load_local(
            db_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )