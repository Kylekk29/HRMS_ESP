import os
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ==================== Logging Settings ====================
LOG_DIR = "./logs"
LOG_FILE = os.path.join(LOG_DIR, f"hr_expert_{datetime.now().strftime('%Y%m%d')}.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ==================== Embedding Settings ====================
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "./AImodels/embedding_model"
)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# ==================== AI Provider Settings ====================
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
MODEL_NORMAL = "deepseek-v4-flash"
MODEL_REASONING = "deepseek-v4-pro"
AI_PROVIDER = os.getenv("AI_PROVIDER", "deepseek")
TIMEOUT = int(os.getenv("TIMEOUT", "120"))         
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))
# ==================== Path Settings ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")
DATABASE_PATH = os.path.join(DATA_ROOT, "HRMS_DATABASE")
NORMAL_DATA_DIR = os.path.join(DATA_ROOT, "NORMAL_DATA")
UPLOAD_DIR = os.path.join(DATABASE_PATH, "uploads")
DB_ROOT = os.path.join(DATA_ROOT, "AIDB", "vector_dbs")
ARCHIVE_DIR = os.path.join(DATA_ROOT, "archive")
METADATA_DIR = os.path.join(DATA_ROOT, "metadata")
HRMS_DATA_DIR = os.path.join(DATA_ROOT, "hrms_data")

# ==================== Vector DB Sub-directories ====================
CV_DB_DIR = os.path.join(DB_ROOT, "cv_candidates")
CULTURE_DB_DIR = os.path.join(DB_ROOT, "company_culture")
EMPLOYEE_DB_DIR = os.path.join(DB_ROOT, "employees")

# ==================== HRMS Files ====================
HRMS_RECORDS_FILE = os.path.join(HRMS_DATA_DIR, "employee_records.json")
HRMS_SCHEMA_VERSION = "1.0"

# ==================== Screening History ====================
SCREENING_HISTORY_DIR = os.path.join(DATA_ROOT, "screening_history")
SCREENING_HISTORY_FILE = os.path.join(SCREENING_HISTORY_DIR, "screening_records.json")

# ==================== Payroll ====================
PAYROLL_RECORDS_DIR = os.path.join(DATA_ROOT, "payroll_data")

# ==================== Ensure all directories exist ====================
for dir_path in [
    UPLOAD_DIR, 
    DB_ROOT, 
    CV_DB_DIR, 
    CULTURE_DB_DIR, 
    EMPLOYEE_DB_DIR,
    ARCHIVE_DIR, 
    METADATA_DIR, 
    HRMS_DATA_DIR, 
    LOG_DIR, 
    SCREENING_HISTORY_DIR,
    PAYROLL_RECORDS_DIR,
    NORMAL_DATA_DIR, 
]:
    os.makedirs(dir_path, exist_ok=True)

# ==================== Version Control Settings ====================
HASH_ALGORITHM = "sha256"
VERSION_TRACKING_ENABLED = True
MAX_VERSIONS_PER_FILE = 10

# ==================== Batch Screening ====================
MAX_CV_BATCH_SIZE = 25
CV_RETRIEVAL_K = 3
CULTURE_RETRIEVAL_K = 3

# ==================== Configure Logging ====================
def setup_logging():
    """Configure root logger with file and console handlers."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


setup_logging()