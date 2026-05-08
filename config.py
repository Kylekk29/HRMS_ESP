import os
from dotenv import load_dotenv

load_dotenv()

# Embedding Settings
EMBEDDING_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "models/text2vec-chinese")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# AI Provider Settings
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Path Settings
UPLOAD_DIR = "./uploads"
DB_ROOT = "./vector_dbs"
DATA_DIR = "./data"

for dir_path in [UPLOAD_DIR, DB_ROOT, DATA_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Security
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
