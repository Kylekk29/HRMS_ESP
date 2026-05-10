# AI-HR Bridge Platform v4.0

An enterprise-grade AI-powered Human Resources Management System (HRMS) that combines traditional HR operations with cutting-edge AI capabilities for recruitment, employee development, and workforce analytics.

## 📋 Overview

AI-HR Bridge Platform is a full-featured HRMS solution that bridges the gap between conventional HR management and modern AI technologies. Built with FastAPI backend and modern vanilla JavaScript frontend, it provides seamless integration of HR operations with AI-driven insights.

## ✨ Features

### Core HR Operations
- **Employee Management** - Full CRUD operations, department hierarchies, KPI tracking
- **Attendance Management** - Check-in/out, daily/monthly summaries, absence tracking
- **Leave Management** - Request workflow with approval/rejection, balance tracking
- **Payroll Processing** - Automatic calculation with deductions for absences, late arrivals, and leave

### AI-Powered Features
- **Intelligent CV Screening** - Batch process up to 25 resumes with customizable scoring weights
- **Company Culture Indexing** - Upload policy documents for RAG-based culture alignment
- **Interview Assistant** - AI evaluation of transcripts across 7 dimensions
- **Employee Chat** - RAG-powered conversation using complete employee data
- **Skill Extraction & Gap Analysis** - Identify employee skills and recommend training
- **Salary Adjustment Suggestions** - Data-driven recommendations based on KPI and tenure

### Technical Highlights
- **Version-Controlled Embeddings** - FAISS vector database with automatic versioning
- **RAG Architecture** - Semantic search across employee documents, CVs, and company policies
- **Multi-Model Support** - Works with DeepSeek, OpenAI-compatible APIs
- **Automatic Retry Logic** - Exponential backoff for API reliability

## 🏗️ Architecture

```
Frontend (Single HTML)
    │
    ▼
FastAPI (main_api.py)
    │
    ├── HRMS Manager (hrms_manager.py)
    ├── Payroll Manager (payroll_manager.py)
    ├── Development Manager (development_manager.py)
    ├── Task Router (task_router.py)
    ├── Embedding Manager (embedding_mgr.py)
    ├── Version Manager (version_manager.py)
    └── AI Provider (model_provider.py)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- 4GB RAM minimum (8GB recommended)
- API key for AI provider (DeepSeek recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-hr-bridge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file from template
cp .env.example .env
# Edit .env with your API key
```

### Configuration

Create `.env` file:

```env
API_KEY=your_deepseek_api_key_here
BASE_URL=https://api.deepseek.com
AI_PROVIDER=deepseek
EMBEDDING_MODEL_NAME=./AImodels/embedding_model
EMBEDDING_DEVICE=cpu
LOG_LEVEL=INFO
```

> **Note:** The platform will automatically download the embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) on first run to `./AImodels/`.

### Running the Platform

```bash
python main_api.py
```

Access the platform at: http://127.0.0.1:8000

## 📁 Project Structure

```
ai-hr-bridge/
├── main_api.py              # FastAPI application (main entry point)
├── index.html               # Frontend dashboard
├── config.py                # Configuration management
├── requirements.txt         # Python dependencies
├── prompts.json             # AI prompt templates
│
├── hrms_manager.py          # Employee records, attendance, leave
├── payroll_manager.py       # Salary calculation engine
├── development_manager.py   # Skills, gaps, course recommendations
├── task_router.py          # Workflow orchestration
├── embedding_mgr.py        # Vector DB operations (FAISS)
├── version_manager.py      # Version tracking for embeddings
├── model_provider.py       # LLM wrapper (DeepSeek/OpenAI)
└── main_handler.py         # Legacy handler (backward compat)

├── data/                   # Application data directory
│   ├── HRMS_DATABASE/      # Uploaded files
│   ├── AIDB/vector_dbs/   # FAISS indices
│   │   ├── cv_candidates/  # CV embeddings
│   │   ├── company_culture/# Policy embeddings
│   │   └── employees/      # Employee document embeddings
│   ├── hrms_data/          # JSON data storage
│   ├── metadata/           # Version tracking
│   ├── logs/               # Application logs
│   └── screening_history/  # CV screening records
│
└── AImodels/               # Downloaded embedding models
```

## 🎯 API Endpoints

### AI Features

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload_culture` | POST | Upload company policy documents |
| `/api/screen_cvs_with_weights` | POST | Batch CV screening with custom weights |
| `/api/interview_assist` | POST | AI interview transcript analysis |
| `/api/employee_chat` | POST | RAG-powered employee Q&A |
| `/api/upload_employee` | POST | Employee document upload |

### HRMS Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/hrms/employees` | GET/POST | List/create employees |
| `/api/hrms/employees/{id}` | GET/PUT/DELETE | Employee CRUD |
| `/api/attendance/checkin` | POST | Record check-in |
| `/api/attendance/checkout` | POST | Record check-out |
| `/api/leave/request` | POST | Submit leave request |
| `/api/leave/approve/{id}` | POST | Approve leave |
| `/api/payroll/{id}` | GET | Calculate monthly salary |

### Development

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/employees/{id}/skills` | GET | Extract employee skills |
| `/api/employees/{id}/skill_gaps` | GET | Gap analysis |
| `/api/employees/{id}/course_recommendations` | GET | Course suggestions |

## 🔧 Key Fixes Implemented

### FIX #1: Atomic Versioning
Version metadata is written **only after** FAISS index is successfully saved to disk. On embedding failure, no version entry is created, allowing clean retry.

### FIX #2: Index Validation
When a version is marked as "reused", the system verifies the FAISS index exists on disk. If missing, it transparently rebuilds without duplicating version metadata.

### FIX #3: Safe ID Generation
`clean_id()` appends short content hash when sanitization changes the string, preventing different raw IDs from mapping to the same filesystem path.

### FIX #4: Temporary File Cleanup
All uploaded files are properly cleaned up even on error conditions.

### FIX #5: API Error Handling
Comprehensive error handling with graceful fallbacks when AI services are unavailable.

### FIX #6: Employee Validation
Document uploads verify employee exists in HRMS before embedding.

### FIX #7: Graceful Degradation
Skill extraction falls back to HRMS data when no documents are embedded.

## 🔌 AI Provider Support

Supports any OpenAI-compatible API endpoint:

- **DeepSeek** (recommended) - `https://api.deepseek.com`
- **OpenAI** - `https://api.openai.com/v1`
- **Local** - Any local LLM with OpenAI-compatible API

Configure via `BASE_URL` and `API_KEY` in `.env`.

## 📊 Performance Characteristics

| Operation | Typical Time |
|-----------|--------------|
| Document embedding (500KB) | 3-8 seconds |
| CV screening (10 resumes) | 15-30 seconds |
| Employee chat response | 2-5 seconds |
| Monthly payroll calculation | <1 second per employee |

## 🐛 Troubleshooting

### Embedding Model Download Fails
The platform uses `sentence-transformers` which may time out on first download. Run:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

### API Connection Issues
Test your AI provider connection:
```bash
python test_ai_api.py --test-all
```

### FAISS Index Corruption
If you encounter `allow_dangerous_deserialization` errors, re-upload the affected document to rebuild the index.

### Log Files
All operations are logged to `./logs/hr_expert_YYYYMMDD.log`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python test_ai_api.py --test-all`
5. Submit a pull request

## 📄 License

Proprietary - All rights reserved.

---

**Version:** 4.0  
**Last Updated:** 2026-05-10  
**Python Version:** 3.9+  
**AI Model:** DeepSeek Chat / DeepSeek Reasoner