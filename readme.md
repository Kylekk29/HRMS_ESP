# AI-HR Bridge Platform v4.0

**Enterprise AI-Powered Human Resources Management System**

A full-stack HRMS solution integrating traditional HR operations (employee records, attendance, leave, payroll) with cutting-edge AI capabilities (CV screening, RAG-powered employee chat, interview analysis, skill extraction, salary adjustment suggestions). Built for 悦途汽车贸易有限公司 (Yuè Tú Automotive Trading Co., Ltd.).

---

## Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Tech Stack](#-tech-stack)
5. [Quick Start](#-quick-start)
6. [Project Structure](#-project-structure)
7. [API Reference](#-api-reference)
8. [Module Deep Dive](#-module-deep-dive)
9. [Data Model](#-data-model)
10. [Configuration](#-configuration)
11. [AI Pipeline](#-ai-pipeline)
12. [Key Fixes](#-key-fixes)
13. [Performance](#-performance)
14. [Troubleshooting](#-troubleshooting)
15. [License](#-license)

---

## Overview

AI-HR Bridge Platform bridges the gap between traditional HR management software and modern AI technologies. It is designed as a **unified single-page application** backed by a **FastAPI REST API**, powered by **DeepSeek AI (via LangChain)** with **FAISS vector databases** for semantic search and RAG (Retrieval-Augmented Generation).

### What It Does

| Category | Capability |
|----------|-----------|
| **Core HR** | Employee CRUD, department hierarchy, KPI tracking, emergency contacts |
| **Attendance** | Check-in/out, auto late detection, overtime computation, daily/monthly reports |
| **Leave** | Request submission, approval workflow, balance tracking, Taiwan labour law compliance |
| **Payroll** | Automatic monthly calculation with deductions, payslip generation, department aggregation |
| **AI Recruitment** | Batch CV screening (up to 25), culture-fit scoring, interview transcript analysis (7 dimensions) |
| **AI Employee Insights** | RAG-powered employee chat, skill extraction, gap analysis, course recommendations |
| **AI Strategy** | Salary adjustment suggestions (KPI + tenure driven), company culture indexing |

---

## Key Features

### Core HR Operations (Modules 1–3)

- **Employee Management** — Full CRUD with JSON persistence, department hierarchy trees, status tracking (Active / On Leave / Terminated), KPI entries, emergency contact storage, position tracking, employment type (Full-time / Part-time / Contract / Intern)
- **Attendance Management (Module 1)** — Check-in/out recording with automatic late detection (threshold: 09:30), real-time work hours calculation, overtime tracking (>8h), half-day detection (<4h), daily attendance summary across all employees, monthly attendance history per employee, manual absent marking with reason
- **Leave Management (Module 2)** — Complete leave request workflow: Submit → Pending → Approve/Reject. Five leave types (Annual, Sick, Personal, Maternity, Special). Automatic balance check with pending request aggregation. Date overlap conflict detection. Smart attendance integration — approved leave auto-marks attendance records and deduplicates conflicts with existing clock records. Taiwan Labour Standards Act compliance (tenure-based annual leave: 3 days <6mo → 7 days <1yr → 10 days <2yr → 14 days <5yr → 15 days <10yr → max 30 days)
- **Payroll Processing (Module 3)** — Automatic monthly salary calculation: base salary + overtime pay (1.5x rate) + bonus - absent deduction (daily rate) - late penalty (2x hourly rate) - half-day deduction (50% daily rate) - leave deduction (sick leave exempted). Formatted payslip text generation. Department-level payroll aggregation with per-employee breakdown. AI-suggested annual salary adjustments based on KPI average score (≥90: 8–12%, 80-89: 5–8%, 70-79: 3–5%, <70: 0–2%) plus tenure bonus (≥3yr: +1%, ≥5yr: +2%)

### AI-Powered Features (Modules 4–5, v4.0)

- **Batch CV Screening** — Upload up to 25 resumes (PDF / DOCX / TXT) simultaneously. AI evaluates each candidate against the job description using 5 dimensions with customizable weights: Core Competency (default 30%), Experience (25%), Education (10%), Culture Fit (15%), Development Potential (20%). Outputs: overall weighted score, match status (Highly Suitable / Partially Suitable / Borderline / Not Suitable), per-dimension scores, strengths, weaknesses, hiring risks, culture alignment analysis, improvement suggestions, and 3 tailored interview focus questions. All results persisted to screening history with full detail retrieval.
- **Company Culture RAG** — Upload company handbook, policy documents, or values statements. Documents are chunked (500 chars, 50 char overlap), embedded via multilingual HuggingFace model (`paraphrase-multilingual-MiniLM-L12-v2`), and stored in FAISS vector index. Version-controlled with SHA-256 content hashing — identical content reuses existing index, modified content creates new version with automatic archival. Used as context during CV screening and interview evaluation for culture-fit assessment.
- **Interview Assistant (Module 4)** — Paste interview transcript + job description → AI evaluates candidate across 7 dimensions: Technical Match, Communication, Logical Thinking, Adaptability, Teamwork, Culture Fit, Learning Agility (each 0–100). Returns: overall score, per-dimension scores, strengths, weaknesses, red flags, key quotes from transcript, follow-up questions, hiring recommendation (Strongly Recommend / Recommend / Hold / Not Recommend), and comprehensive overall assessment in Chinese.
- **Employee AI Chat (v4.0)** — RAG-powered conversational AI about any specific employee. Pulls context from: HRMS basic data (name, position, department, status, hire date, employment type), salary info, leave balance (all types), emergency contact details, KPI history (last 5 entries), attendance records (current month), leave request history, and embedded documents from vector database (CV, profile). Returns data-driven, actionable responses. Pre-built quick actions: Risk Check, Skills Review, Attendance Check.
- **Employee Document Embedding** — Upload CV/Resume, Profile/Bio, Performance Review, or custom documents per employee. Text-based profile editing also supported. Documents auto-classified via AI. Stored in employee-specific FAISS vector databases (`employees/<id>/cv/`, `employees/<id>/profile/`). Enables semantic search for skill extraction and employee chat.
- **Skill Extraction & Development (Module 5)** — Extracts skills from 40+ keyword taxonomy spanning technical (Python, Java, Docker, AWS, etc.), data (SQL, PowerBI, Tableau), project management (Agile, Scrum, PMP), soft skills (Leadership, Communication), HR/Finance, and languages (Mandarin, English, Japanese, Spanish). Proficiency level detection via signal words (beginner/intermediate/advanced). Searches vector DB for skills in uploaded CVs/profiles; gracefully falls back to HRMS records if no documents are embedded.

### System Features

- **Enhanced Dashboard** — Real-time KPIs: total/active/on leave/terminated employees, new hires this month, turnover rate, today's attendance breakdown (present/late/half-day/absent/on leave), pending leave approvals count, monthly payroll estimate (total + average), department summary, recent screening history
- **Internal Marketplace (Module 7)** — Project posting, employee-project skill matching, application submission with approval/rejection workflow
- **Health Check** — `/api/health` endpoint for monitoring

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     index.html (2979 lines)                   │
│          Single-Page Application — Vanilla JavaScript         │
│           DM Sans + Syne fonts · Chart.js · CSS vars          │
├──────────────────────────┬───────────────────────────────────┤
│   14 UI Sections         │   Responsive Layout (260px sidebar │
│   ├─ Dashboard           │   + fluid main)                    │
│   ├─ CV Screening        │   Modal dialogs for CRUD           │
│   ├─ Company Culture     │   Drag-drop file upload            │
│   ├─ Interview Assist    │   Tab-based employee form          │
│   ├─ Employee Chat       │   Score bar visualization          │
│   ├─ Employee Docs       │   Chart.js radar/bar charts        │
│   ├─ Employees (HRMS)    │   Filter + search tables           │
│   ├─ Attendance          │   Leave balance track bars         │
│   ├─ Leave Management    │   Chat-style UI for AI questions   │
│   ├─ Payroll             │   Screening history table          │
│   └─ Development         │   Status pills (color-coded)       │
└──────────────────────────┴───────────────────────────────────┘
                              │
                    HTTP REST (POST/GET/PUT/DELETE)
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│                     main_api.py (861 lines)                    │
│                  FastAPI Application (v0.136.1)                │
│                        86 API Endpoints                        │
│                  CORS: allow all origins (*)                   │
├──────────┬──────────┬──────────┬──────────┬──────────────────┤
│          │          │          │          │                  │
▼          ▼          ▼          ▼          ▼                  │
┌──────┐ ┌──────┐ ┌───────┐ ┌──────────┐ ┌─────────────────┐ │
│HRMS  │ │Payroll│ │Task   │ │Development│ │Embedding        │ │
│Mgr   │ │Mgr    │ │Router │ │Manager   │ │Manager          │ │
├──────┤ ├───────┤ ├───────┤ ├──────────┤ ├─────────────────┤ │
│- CRUD│ │- Salary│ │- CV   │ │- Skills  │ │- Doc loading    │ │
│- Att │ │  calc │ │  screen│ │- Gap     │ │  (PDF/DOCX/TXT) │ │
│- Leave│ │- Pay- │ │- Culture│ │  analysis│ │- Text splitting │ │
│- Dept │ │  slip │ │  upload│ │- Course  │ │- FAISS build    │ │
│  tree│ │- Adjust│ │- Chat │ │  recs    │ │- Version control│ │
│- KPI │ │  ment │ │- Inter-│ │          │ │- Batch embed    │ │
│      │ │- Dept │ │  view  │ │          │ │- DB load/search │ │
│      │ │  agg  │ │  assist│ │          │ │                 │ │
└──┬───┘ └───┬───┘ └───┬───┘ └────┬─────┘ └───────┬─────────┘ │
   │         │         │           │               │           │
   │         │         │    ┌──────┴──────┐        │           │
   │         │         │    │             │        │           │
   │         │         │    ▼             ▼        │           │
   │         │         │ ┌──────────┐ ┌───────┐    │           │
   │         │         │ │AI Model  │ │Version│    │           │
   │         │         │ │Provider  │ │Mgr    │    │           │
   │         │         │ ├──────────┤ ├───────┤    │           │
   │         │         │ │- DeepSeek│ │- SHA  │    │           │
   │         │         │ │  Chat/   │ │  256  │    │           │
   │         │         │ │  Reasoner│ │  hash │    │           │
   │         │         │ │- LangChn │ │- Meta │    │           │
   │         │         │ │  OpenAI  │ │  data │    │           │
   │         │         │ │- Auto    │ │- Arch │    │           │
   │         │         │ │  retry   │ │  ival │    │           │
   │         │         │ │- JSON    │ └───────┘    │           │
   │         │         │ │  extract │              │           │
   │         │         │ └──────────┘              │           │
   │         │         │                           │           │
   └─────────┴─────────┴───────────────────────────┘           │
                              │                                  │
┌─────────────────────────────▼──────────────────────────────────┤
│                        DATA LAYER                              │
│                                                                │
│  data/hrms_data/              data/AIDB/vector_dbs/            │
│  ├─ employee_records.json     ├─ cv_candidates/<db_id>/        │
│  ├─ attendance_records.json   │   ├─ index.faiss               │
│  ├─ leave_requests.json       │   └─ index.pkl                 │
│  ├─ development_courses.json  ├─ company_culture/<db_id>/      │
│  └─ development_plans.json    │   ├─ index.faiss               │
│                               │   └─ index.pkl                 │
│  data/payroll_data/           └─ employees/<id>/               │
│  └─ payroll_records.json          ├─ cv/<db_id>/               │
│                                   │   ├─ index.faiss           │
│  data/screening_history/          │   └─ index.pkl             │
│  └─ screening_records.json        └─ profile/<db_id>/          │
│                                       ├─ index.faiss           │
│  data/metadata/                      └─ index.pkl             │
│  └─ <file_id>_meta.json                                        │
│                                                                │
│  data/archive/                AImodels/embedding_model/        │
│  └─ archived source files     └─ paraphrase-multilingual-      │
│                                   MiniLM-L12-v2                │
└────────────────────────────────────────────────────────────────┘
```

### Module Dependency Graph

```
config.py ──────────────────────────────────────────────────────┐
  │ (env vars, paths, logging, constants)                       │
  ├──► hrms_manager.py       (uses config paths)                │
  ├──► payroll_manager.py    (uses config paths, imports HRMS)  │
  ├──► version_manager.py    (uses config paths)                │
  ├──► model_provider.py     (uses config API keys + model)     │
  ├──► embedding_mgr.py      (uses config paths + model)        │
  │      └──► version_manager.py                                │
  ├──► task_router.py        (uses config paths)                │
  │      ├──► model_provider.py                                 │
  │      └──► embedding_mgr.py                                  │
  ├──► development_manager.py (uses config paths)               │
  │      ├──► embedding_mgr.py (optional — can work standalone) │
  │      └──► model_provider.py (optional)                      │
  └──► main_api.py           (uses config, all managers)        │
         ├──► hrms_manager.py                                   │
         ├──► payroll_manager.py                                │
         ├──► task_router.py                                    │
         └──► development_manager.py                            │
```

### Data Flow — CV Screening

```
1. User uploads CVs (PDF/DOCX/TXT) + enters JD via UI
2. main_api.py validates inputs, saves files temporarily
3. task_router.batch_screen_cvs() is called
4. For each CV:
   a. embedding_mgr.embed_file_with_versioning()
      i.   version_manager.check_version() — SHA-256 hash check
      ii.  If identical content exists → reuse FAISS index
      iii. If new/updated → load document, split into 500-char chunks
           with 50-char overlap, generate embeddings via HuggingFace
           model, save FAISS index, commit version metadata
5. Retrieve culture context from company_culture FAISS index
6. Build comprehensive context string from all CV vector DBs
7. model_provider.AIModelProvider.cv_screening_ai()
   a. Load prompt from prompts.json → "cv_screening"
   b. Truncate context if >180K chars
   c. Call DeepSeek Chat API via LangChain ChatOpenAI
   d. Extract JSON from response (4 fallback strategies)
   e. Retry up to 3 times with exponential backoff
8. Results saved to screening_history/screening_records.json
9. Return ranked candidates with per-dimension scores, analysis, interview questions
```

---

## Tech Stack

### Backend

| Component | Technology | Version |
|-----------|-----------|---------|
| **Language** | Python | 3.12 |
| **Web Framework** | FastAPI | 0.136.1 |
| **ASGI Server** | Uvicorn | 0.46.0 |
| **LLM Framework** | LangChain | 1.2.17 |
| **LLM SDK** | LangChain-OpenAI | 1.2.1 |
| **LLM Provider** | DeepSeek Chat / DeepSeek Reasoner | API |
| **Embeddings** | HuggingFace sentence-transformers | 5.4.1 |
| **Embedding Model** | paraphrase-multilingual-MiniLM-L12-v2 | local |
| **Vector Database** | FAISS (faiss-cpu) | 1.13.2 |
| **Document Loaders** | PyPDF, TextLoader, Docx2txtLoader | — |
| **Text Splitting** | RecursiveCharacterTextSplitter | LangChain |
| **Data Validation** | Pydantic | 2.13.4 |
| **Environment** | python-dotenv | 1.2.2 |
| **Language Det.** | langdetect | — |

### Frontend

| Component | Technology |
|-----------|-----------|
| **Language** | Vanilla JavaScript (ES6+) |
| **UI Framework** | None — custom CSS with CSS variables design system |
| **Charts** | Chart.js (CDN) |
| **Fonts** | DM Sans (body) + Syne (headings), Google Fonts |
| **Icons** | Unicode emoji |
| **Responsive** | CSS Grid + Flexbox, 900px mobile breakpoint |

### Storage & Infrastructure

| Type | Technology |
|------|-----------|
| **Employee Records** | JSON file (`employee_records.json`) |
| **Attendance** | JSON file (`attendance_records.json`) |
| **Leave Requests** | JSON file (`leave_requests.json`) |
| **Payroll** | JSON file (`payroll_records.json`) |
| **Screening History** | JSON file (`screening_records.json`) |
| **Vector Data** | FAISS binary indexes (`.faiss` + `.pkl`) |
| **Version Metadata** | JSON per file-ID (`<id>_meta.json`) |
| **File Archives** | Original files in `data/archive/` |
| **Logging** | Daily rotated log files (`logs/hr_expert_YYYYMMDD.log`) |

---

## Quick Start

### Prerequisites

- **Python** 3.9+ (developed on 3.12)
- **RAM** 4GB minimum, 8GB recommended (for embedding model)
- **Disk** ~2GB free (for embedding model download + FAISS indices)
- **API Key** — DeepSeek API key (get one at platform.deepseek.com)
- **Network** — Outbound HTTPS to `api.deepseek.com`

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ESG_HRMS

# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key
echo API_KEY=sk-your-deepseek-key-here > .env
echo BASE_URL=https://api.deepseek.com >> .env
```

### First Run

The embedding model will be downloaded automatically on first startup (~120MB):

```bash
python main_api.py
```

Access: **http://127.0.0.1:8000**

### Verify Installation

```bash
# Test AI API connection
python test_ai_api.py --quick

# Run full test suite
python test_ai_api.py --test-all --verbose
```

---

## Project Structure

```
ESG_HRMS/
├── main_api.py                  # FastAPI application (86 endpoints, entry point)
├── index.html                   # Frontend SPA (2979 lines, all UI)
├── config.py                    # Environment, paths, constants, logging setup
├── requirements.txt             # Python dependencies (97 packages)
├── prompts.json                 # AI prompt templates (3 features)
├── test_ai_api.py               # AI API connection test suite (5 tests)
├── readme.md                    # This file
│
├── hrms_manager.py              # Employee CRUD, attendance, leave workflows
├── payroll_manager.py           # Salary calculation engine
├── development_manager.py       # Skill extraction, gap analysis, course recommendations
├── task_router.py               # Workflow orchestration (CV screening, chat, interview)
├── embedding_mgr.py             # Document loading, chunking, embedding, FAISS CRUD
├── version_manager.py           # SHA-256 version control for embeddings
├── model_provider.py            # LangChain LLM wrapper (DeepSeek via OpenAI API)
├── main_handler.py              # Legacy single-request handler (backward compat)
│
├── data/                        # All application data (gitignored)
│   ├── HRMS_DATABASE/uploads/   # Temporary uploaded files
│   ├── AIDB/vector_dbs/         # FAISS vector indexes
│   │   ├── cv_candidates/       # CV embeddings (per batch)
│   │   ├── company_culture/     # Company policy embeddings
│   │   └── employees/           # Per-employee document embeddings
│   ├── hrms_data/               # JSON data stores
│   │   ├── employee_records.json
│   │   ├── attendance_records.json
│   │   ├── leave_requests.json
│   │   ├── development_courses.json
│   │   └── development_plans.json
│   ├── payroll_data/            # Payroll calculation records
│   ├── screening_history/       # CV screening result archives
│   ├── metadata/                # Version tracking per document
│   ├── archive/                 # Archived source file snapshots
│   └── logs/                    # Daily application logs
│
├── AImodels/embedding_model/    # Downloaded embedding model (auto)
├── logs/                        # Application logs
├── .env                         # API keys and environment (gitignored)
└── venv/                        # Python virtual environment
```

---

## API Reference

### AI Features

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| POST | `/api/upload_culture` | Upload & index company policy docs | `file` (PDF/DOCX/TXT) |
| POST | `/api/screen_cvs` | Batch CV screening (default weights) | `jd` (text), `files` (multiple) |
| POST | `/api/screen_cvs_with_weights` | Batch CV screening (custom weights) | `jd`, `files`, `weights_json` |
| POST | `/api/employee_chat` | RAG-powered employee Q&A | `employee_id`, `query`, `conversation_history` |
| POST | `/api/upload_employee` | Upload & embed employee document | `employee_id`, `doc_type`, `file` |
| POST | `/api/interview_assist` | AI interview transcript analysis | `transcript`, `jd`, `competency` |

### HRMS — Employee Records

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/hrms/employees` | List employees (optional `department`, `status` filters) |
| POST | `/api/hrms/employees` | Create new employee |
| GET | `/api/hrms/employees/{id}` | Get employee by ID |
| PUT | `/api/hrms/employees/{id}` | Update employee |
| DELETE | `/api/hrms/employees/{id}` | Delete employee |
| POST | `/api/hrms/employees/{id}/kpi` | Add KPI entry |
| POST | `/api/hrms/employees/{id}/leave` | Direct leave deduction (legacy) |
| POST | `/api/hrms/employees/{id}/edit_profile` | Create/update employee profile (text) |
| PUT | `/api/hrms/employees/{id}/department` | Update department assignment |

### HRMS — Departments

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/hrms/departments/summary` | Department headcount + salary totals |
| GET | `/api/hrms/departments/tree` | Hierarchical department tree with employees |

### Attendance (Module 1)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/attendance/checkin` | Record check-in (auto late detection >09:30) |
| POST | `/api/attendance/checkout` | Record check-out (computes work hours + overtime) |
| GET | `/api/attendance/monthly/{id}` | Monthly attendance records (year, month params) |
| GET | `/api/attendance/daily` | Daily attendance summary (date param) |
| POST | `/api/attendance/absent` | Mark employee absent on a date |

### Leave Management (Module 2)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/leave/request` | Submit leave request for approval |
| POST | `/api/leave/approve/{request_id}` | Approve pending leave (with attendance sync) |
| POST | `/api/leave/reject/{request_id}` | Reject pending leave |
| GET | `/api/leave/pending` | List pending requests (optional `department` filter) |
| GET | `/api/leave/all` | All requests (optional `employee_id` filter) |
| GET | `/api/leave/summary/{employee_id}` | Leave balance summary per type |

### Payroll (Module 3)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/payroll/{employee_id}` | Monthly salary calculation with full breakdown |
| GET | `/api/payroll/{employee_id}/payslip` | Formatted text payslip |
| GET | `/api/payroll/department/{department}` | Department-level payroll aggregation |
| GET | `/api/payroll/{employee_id}/adjustment` | AI salary adjustment suggestion |

### Employee Development (Module 5)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/employees/{id}/skills` | Extract employee skills from HRMS + vector DB |
| GET | `/api/employees/{id}/skill_gaps` | Skill gap analysis vs target role |
| GET | `/api/employees/{id}/course_recommendations` | Recommended training courses |
| POST | `/api/employees/{id}/development_plan` | Create development plan |
| GET | `/api/employees/{id}/development_plan` | Get development plan |

### Internal Marketplace (Module 7)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/marketplace/projects` | Create new project |
| GET | `/api/marketplace/projects` | List all projects |
| GET | `/api/marketplace/projects/{id}/matches` | Find matching employees |
| POST | `/api/marketplace/projects/{id}/apply` | Apply for project |
| POST | `/api/marketplace/applications/{id}/approve` | Approve application |
| POST | `/api/marketplace/applications/{id}/reject` | Reject application |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve frontend HTML |
| GET | `/api/dashboard` | Enhanced dashboard with all KPIs |
| GET | `/api/health` | Health check |
| GET | `/api/screening_history` | Paginated CV screening history |
| GET | `/api/screening_history/{id}` | Full screening detail |

---

## Module Deep Dive

### 1. Embedding Manager (`embedding_mgr.py` — 441 lines)

Handles document loading, text chunking, embedding generation, and FAISS persistence.

**Document Loading:**
- PDF → `PyPDFLoader`
- DOCX → `Docx2txtLoader`
- TXT → `TextLoader` (UTF-8)

**Text Chunking:**
- Splitter: `RecursiveCharacterTextSplitter`
- Chunk size: 500 characters
- Overlap: 50 characters
- Separators: `\n\n`, `\n`, `。`, `！`, `？`, `，`, ` `, ``

**Embedding Model:**
- `paraphrase-multilingual-MiniLM-L12-v2`
- Device: CPU (configurable via `EMBEDDING_DEVICE`)
- Normalized embeddings enabled

**DB Layout:**
```
vector_dbs/
  cv_candidates/<db_id>/       — CVs from batch screening
  company_culture/<db_id>/     — Uploaded policy documents
  employees/<clean_id>/cv/<db_id>/         — Employee CV
  employees/<clean_id>/profile/<db_id>/    — Employee profile
```

**Key Methods:**
- `embed_file_with_versioning()` — Embeds a file with version control; checks SHA-256 hash first, skips if identical content exists; if index missing on disk, repairs transparently
- `embed_employee_document()` — Embeds a document under employee's folder with doc_type
- `embed_cv_batch()` — Processes up to 25 CVs concurrently
- `load_db()` — Loads a FAISS index from disk
- `load_employee_db()` — Loads latest employee document index, validates index exists on disk
- `clean_id()` — Sanitizes IDs for filesystem safety (appends MD5 suffix when characters were replaced)
- `get_version_history()` — Returns version history for any file

### 2. Version Manager (`version_manager.py` — 339 lines)

Atomic two-phase version control system.

**Phase 1 — `check_version()` (read-only):**
1. Calculate SHA-256 hash of the file
2. Search existing metadata for matching hash
3. If match found → validate FAISS index exists on disk
4. If index missing → signal "repaired" action (rebuild without new metadata)
5. If no match → generate proposed db_id and version number
6. Returns dict with `is_new`, `action`, `db_id`, `version_number`, `_internal`

**Phase 2 — `commit_new_version()` (write):**
1. Only called AFTER FAISS index successfully saved to disk
2. Archives the source file to `data/archive/`
3. Appends version entry to metadata JSON
4. Prunes old versions (keeps max 10 per file)
5. Updates `current_hash` and `current_db_id` pointers

**FIX #1 (Atomicity):** If embedding fails, metadata is never written — re-uploading triggers a clean embed.

**FIX #2 (Integrity):** If FAISS index is deleted from disk but metadata still references it, system detects the gap and triggers a repair rebuild.

### 3. AI Model Provider (`model_provider.py` — 276 lines)

LangChain wrapper around DeepSeek (OpenAI-compatible) API.

**Capabilities:**
- Context truncation (>180K chars → smart head+tail preservation)
- JSON extraction with 4 fallback strategies (direct parse → regex object → regex array → common error fix)
- Automatic retry with exponential backoff (2^attempt, capped at 30s)
- Rate limit handling (5s * attempt, capped at 60s)
- Error classification (Connection, Timeout, RateLimit, API, Unexpected)

**Features:**
- `cv_screening_ai()` — General-purpose JSON-structured AI call (used for CV screening, interview analysis, document classification)
- `chat()` — Raw text response (used for employee chat)

**AI Features Mapped:**

| Feature | Method | Response Format |
|---------|--------|----------------|
| CV Screening | `cv_screening_ai("cv_screening")` | JSON with candidate results array |
| Interview Analysis | `cv_screening_ai("interview_assistant")` | JSON with 7-dimension scores |
| Employee Chat | `chat("employee_chat")` | Plain text |
| Document Classification | `cv_screening_ai("document_classifier")` | JSON classification |

**Prompt Templates** (from `prompts.json`):
- `cv_screening` — Senior HR Director persona, 5-axis evaluation, weighted scores, detailed analysis with strengths/weaknesses/risks/interview questions
- `employee_chat` — AI HR Assistant persona, data-driven, actionable insights, confidentiality-aware
- `interview_assistant` — Expert interviewer persona (Chinese), 7-dimension evaluation, key quotes, follow-up questions

### 4. Task Router (`task_router.py` — 574 lines)

Central orchestration layer that coordinates all AI workflows.

**Workflows:**
1. **Company Culture Upload** — Delegates to `EmbeddingManager.embed_file_with_versioning()` with category "company_culture"
2. **Batch CV Screening** — (a) Embed all CVs → (b) Retrieve culture context → (c) Build candidate context strings from vector DB → (d) Call AI with all candidates + JD + weights → (e) Attach metadata → (f) Save screening history
3. **Employee Chat** — (a) Gather HRMS data → (b) Search vector DB for embedded documents → (c) Build comprehensive RAG context → (d) Call AI with context + query → (e) Return response with RAG status
4. **Interview Analysis** — (a) Validate inputs → (b) Call AI with transcript + JD + competency requirements → (c) Parse and return dimension scores

### 5. HRMS Manager (`hrms_manager.py` — 683 lines)

Core HR data layer with three JSON persistence stores.

**Employee Records:**
- CRUD operations with JSON file persistence
- Employment types: Full-time, Part-time, Contract, Intern
- Statuses: Active, On Leave, Terminated
- KPI entries with score, rating, period, comments
- Emergency contact storage
- Department hierarchy tree (supports > delimited paths)

**Attendance (Module 1):**
- Check-in with duplicate prevention and auto-late detection (>09:30)
- Check-out with work hours calculation and overtime detection (>8h)
- Half-day status when work hours <4h
- Monthly attendance query by year/month
- Daily summary across all employees
- Absent marking with reason

**Leave Management (Module 2):**
- Five leave types with total/used balance tracking
- Tenure-based annual leave calculation (Taiwan Labour Standards Act):
  - <6 months: 3 days
  - 6-12 months: 7 days
  - 1-2 years: 10 days
  - 2-5 years: 14 days
  - 5-10 years: 15 days
  - 10+ years: 15 + (years-10), max 30 days
- Request submission with balance validation (aggregates pending requests)
- Date overlap detection with existing approved/pending requests
- Approval workflow with automatic attendance marking and conflict handling
- Rejection with reason tracking
- Leave balance summary query

### 6. Payroll Manager (`payroll_manager.py` — 384 lines)

Monthly salary calculation engine.

**Salary Formula:**
```
Gross = Base + OvertimePay + Bonus - AbsentDeduction - LatePenalty - HalfDayDeduction - LeaveDeduction

Where:
  DailyRate     = Base / WorkingDaysInMonth
  HourlyRate    = DailyRate / 8
  OvertimePay   = OvertimeHours × HourlyRate × 1.5
  AbsentDed     = AbsentDays × DailyRate
  LatePenalty   = LateDays × HourlyRate × 2
  HalfDayDed    = HalfDays × DailyRate × 0.5
  LeaveDed      = DeductibleLeaveDays × DailyRate  (sick leave exempted)
```

**Key Features:**
- Working days computed from actual calendar (Mon–Fri count)
- Attendance stats aggregated with mutual exclusion (Absent, On Leave, Late, Half-day, Present)
- Deductible leave: annual_leave, personal_leave, special_leave (sick leave is NOT deducted)
- Formatted payslip generation in plain text with earnings, deductions, and attendance summary
- Department-level aggregation with per-employee detail
- Annual adjustment suggestions: KPI-based range (≥90: 8–12%, 80-89: 5–8%, 70-79: 3–5%, <70: 0–2%) + tenure bonus (≥3yr: +1%, ≥5yr: +2%)

### 7. Development Manager (`development_manager.py` — 178 lines)

Employee skill extraction and development planning.

**Skill Taxonomy (40+ keywords):**
- **Programming:** Python, Java, JavaScript, TypeScript, Go, Rust, C++, C#
- **Data:** SQL, MySQL, PostgreSQL, MongoDB, Redis, Elasticsearch
- **DevOps:** Docker, Kubernetes, Terraform, Ansible, Jenkins
- **Cloud:** AWS, Azure, GCP
- **AI/ML:** Machine Learning, Deep Learning, TensorFlow, PyTorch, NLP
- **Frontend:** React, Vue, Angular, Node.js, FastAPI, Django, Flask
- **BI:** PowerBI, Tableau, Excel, VLOOKUP, DAX
- **PM:** Project Management, PMP, Agile, Scrum, Kanban
- **Soft:** Leadership, Communication, Teamwork, Presentation
- **Domain:** HR, Recruitment, Payroll, Labour Law, Finance, Accounting, Sales, Marketing, SEO, CRM, Salesforce
- **Languages:** Mandarin, English, Japanese, Spanish

**Proficiency Detection (via signal words):**
- **Advanced:** expert, senior, lead, principal, advanced, 專家, 高級, 資深, 精通
- **Intermediate:** proficient, experienced, 3+ years, 熟練, 中級
- **Beginner:** familiar, basic, entry, junior, 初級, 了解, 基礎

**Data Sources (priority order):**
1. Vector DB — search embedded CV/profile documents for skills
2. HRMS fallback — position, notes, KPI comments (limited accuracy)

**FIX #7 (Graceful Degradation):**
- `FileNotFoundError` → logged as INFO (normal — no document uploaded yet)
- Other exceptions → logged as ERROR with repair instructions (possibly corrupted FAISS index)

### 8. Frontend (`index.html` — 2979 lines)

Single-page application with 14 UI sections:

| Section | Key UI Elements |
|---------|----------------|
| Dashboard | 8 stat cards, screening history table |
| CV Screening | JD textarea, 5-slider weight panel, drag-drop file zone, Chart.js score chart, detailed candidate cards with score bars and analysis grids |
| Company Culture | Drag-drop upload zone |
| Interview Assist | JD textarea, competency input, transcript textarea, results with 7-dimension radar chart |
| Employee Chat | Employee selector with quick info panel, chat history display, question input, 4 quick-action buttons (Risk Check, Skills Review, Attendance, Clear) |
| Employee Docs | Employee ID + doc type selector + file upload |
| Employees (HRMS) | Stat cards, searchable/filterable table, modal with 5 tabs (Basic, Salary, Leave, KPI, Profile) |
| Attendance | Check-in/out buttons, daily summary date picker, monthly records view |
| Leave Management | Leave request form, pending approvals list, balance checker |
| Payroll | Salary calculation form, payslip viewer, adjustment suggestion |
| Development | Employee ID + target role inputs, skill extraction button |

**Design System:**
- CSS variables for theming (--ink, --surface, --accent, --green, --red, --amber, --teal)
- Component library: cards, stat cards, score bars, pills, alerts, modals, dropzones, tab pills, form grids
- Responsive: sidebar collapses to icon-only at 900px, form grids go single-column
- Animations: fadeUp on section change, spin on loader, hover transitions

---

## Data Model

### Employee Record

```json
{
  "employee_id": "EMP001",
  "full_name": "John Doe",
  "email": "john@example.com",
  "phone": "+852-1234-5678",
  "department": "Engineering/Backend",
  "position": "Senior Developer",
  "employment_type": "Full-time",
  "hire_date": "2024-01-15",
  "status": "Active",
  "salary": {
    "base": 50000,
    "currency": "HKD",
    "pay_cycle": "Monthly",
    "last_review": "2025-06-01",
    "bonus": 5000
  },
  "leave": {
    "annual_leave_total": 14,
    "annual_leave_used": 3,
    "sick_leave_total": 30,
    "sick_leave_used": 1,
    "personal_leave_total": 14,
    "personal_leave_used": 0,
    "maternity_leave_total": 0,
    "maternity_leave_used": 0,
    "special_leave_total": 0,
    "special_leave_used": 0
  },
  "kpi": [
    {
      "period": "2025-Q3",
      "score": 92,
      "rating": "Excellent",
      "comments": "Exceeded all sprint goals",
      "added_at": "2025-10-01T10:00:00"
    }
  ],
  "emergency_contact": {
    "name": "Jane Doe",
    "relationship": "Spouse",
    "phone": "+852-8765-4321"
  },
  "notes": "Key contributor to cloud migration project",
  "created_at": "2024-01-15T09:00:00",
  "updated_at": "2025-10-01T10:00:00"
}
```

### Attendance Record

```json
{
  "id": "ATT20260515_001",
  "employee_id": "EMP001",
  "date": "2026-05-15",
  "check_in": "09:15",
  "check_out": "18:30",
  "work_hours": 8.5,
  "status": "Present",
  "overtime_hours": 0.5,
  "notes": ""
}
```

### Leave Request

```json
{
  "request_id": "LR20260515_001",
  "employee_id": "EMP001",
  "employee_name": "John Doe",
  "department": "Engineering",
  "leave_type": "annual_leave",
  "start_date": "2026-06-01",
  "end_date": "2026-06-03",
  "days": 3,
  "reason": "Family vacation",
  "status": "Pending",
  "submitted_at": "2026-05-15T14:30:00",
  "approved_by": null,
  "approved_at": null,
  "rejection_reason": null
}
```

---

## Configuration

All settings in `config.py` + `.env` file:

### Environment Variables (`.env`)

```env
# Required
API_KEY=sk-your-deepseek-api-key

# Optional (defaults shown)
BASE_URL=https://api.deepseek.com
AI_PROVIDER=deepseek
EMBEDDING_MODEL_NAME=./AImodels/embedding_model
EMBEDDING_DEVICE=cpu
LOG_LEVEL=INFO
TIMEOUT=120
MAX_RETRIES=5
RETRY_DELAY=5
```

### Config Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `CHUNK_SIZE` | 500 | Characters per text chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `MAX_CV_BATCH_SIZE` | 25 | Max resumes per screening batch |
| `CV_RETRIEVAL_K` | 3 | Chunks retrieved per CV |
| `CULTURE_RETRIEVAL_K` | 3 | Chunks retrieved from culture DB |
| `MAX_VERSIONS_PER_FILE` | 10 | Version history depth |
| `HASH_ALGORITHM` | sha256 | File content hashing |
| `MODEL_NORMAL` | deepseek-chat | Default AI model |
| `MODEL_REASONING` | deepseek-reasoner | Reasoning model |

---

## AI Pipeline

### Prompt Architecture

All prompts stored in `prompts.json` with two-part structure:
- `system_prompt` — AI persona and behavior rules
- `user_template` — Input variables: `{ctx}`, `{jd}`, `{query}`, `{context}`, `{culture_ctx}`, `{weights_text}`, `{competency}`

### CV Screening Scoring Formula

```
overall_score = (
    core_competency_match × w_core +
    experience_match × w_exp +
    education_match × w_edu +
    culture_fit_score × w_culture +
    development_potential × w_dev
) / 100

Weights are user-customizable and auto-normalized to sum to 100
Default: core=30, exp=25, edu=10, culture=15, dev=20
```

### RAG Pipeline (Employee Chat)

1. **Structured data** — Load from JSON: HRMS record, salary, leave balances, emergency contact, KPI history (last 5), attendance (current month), leave requests (last 5)
2. **Vector data** — `similarity_search(query, k=3)` across employee CV and profile FAISS DBs
3. **Context assembly** — All sources concatenated with section headers
4. **AI invocation** — Full context passed to `chat()` with `employee_chat` prompt
5. **Response** — Plain text, data-driven, with `rag_used` flag indicating whether vector data contributed

### Error Resilience

The AI model provider implements:
- **3 retries** with exponential backoff (2^attempt seconds)
- **Rate limit** backoff (5 × attempt seconds, max 60s)
- **Timeout** handling (connection/timeout errors → retry, API errors → fail fast)
- **Context truncation** — automatically truncates inputs >180K characters (preserves 60% head + 40% tail)
- **JSON extraction** with 4 fallback strategies for robust parsing

---

## Key Fixes

### FIX #1 — Atomic Versioning
Version metadata is committed **only after** FAISS index is successfully persisted to disk. If embedding fails mid-process, no version entry exists — re-uploading the same file triggers a clean embed.

### FIX #2 — Index Integrity Validation
When `check_version()` finds a matching hash (reuse), it verifies the FAISS index file exists on disk. If missing (e.g., deleted by operator), action is set to `"repaired"` — the index is transparently rebuilt without creating duplicate metadata.

### FIX #3 — Safe ID Generation
`clean_id()` sanitizes strings for filesystem use. If sanitization changes the original string (e.g., `"user@123"` → `"user_123"`), a 6-char MD5 suffix is appended (`"user_123_a3f9c1"`) so different raw IDs never collide.

### FIX #4 — Temporary File Cleanup
All uploaded files are cleaned up via `_cleanup()` in `finally` blocks, even on error conditions.

### FIX #5 — Graceful API Error Handling
All AI endpoints return structured error responses (`{success: false, error: "..."}`) instead of throwing unhandled exceptions. HRMS operations validate employee existence before proceeding.

### FIX #6 — Employee Existence Validation
Document uploads and chat requests verify the employee exists in HRMS before embedding or querying, preventing orphaned vector data.

### FIX #7 — Graceful Skill Extraction Degradation
Skill extraction distinguishes between expected absence (no document uploaded — INFO log) and unexpected errors (corrupted FAISS index — ERROR log with repair instructions). Falls back to HRMS text data when vector DB is unavailable.

### FIX #8 — F-String Template Escaping in Prompts
The `interview_assistant.system_prompt` in `prompts.json` contained unescaped single braces `{`/`}` for the JSON response example. LangChain's f-string template parser interpreted these as nested replacement fields, causing `Invalid format specifier` errors. Fixed by escaping JSON braces with `{{`/`}}`.

---

## Performance

| Operation | Typical Time | Factors |
|-----------|-------------|---------|
| Document embedding (500KB PDF) | 3–8 seconds | File size, CPU speed |
| CV batch screening (10 resumes) | 15–30 seconds | CV count, AI API latency |
| CV batch screening (25 resumes) | 30–60 seconds | CV count, chunk volume |
| Employee chat response | 2–5 seconds | Context size, AI model |
| Interview analysis | 8–15 seconds | Transcript length |
| Monthly payroll (1 employee) | <1 second | Pure computation |
| FAISS index load | 1–3 seconds | Index file size |
| Company culture upload | 5–15 seconds | Document size |

### Optimizations
- **Version control** — Same-content files reuse existing indices (instant skip)
- **Smart truncation** — Preserves head (60%) and tail (40%) when context exceeds limits
- **Concurrent CV embedding** — Files processed sequentially but each benefits from caching
- **JSON local storage** — No database overhead for HRMS data
- **CSS variables** — Zero-runtime theming
- **No frontend framework** — Minimal bundle size, no build step

---

## Troubleshooting

### First Run — Embedding Model Download
The `sentence-transformers` library downloads the embedding model on first use. If the download times out:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

### AI API Connection
Test connectivity:
```bash
python test_ai_api.py --quick          # Basic connection test
python test_ai_api.py --test-all       # Full 5-test suite
```

### FAISS Index Corruption
If you encounter errors about `allow_dangerous_deserialization`, the FAISS index may be corrupted. Re-upload the affected document to rebuild the index automatically.

### Logs
All operations logged to `./logs/hr_expert_YYYYMMDD.log`. Check for detailed error context.

### Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| "API key not found" | Missing `.env` file | Create `.env` with `API_KEY=sk-...` |
| "Unsupported file type" | Wrong extension | Only PDF, DOCX, TXT supported |
| "Employee not found" | ID doesn't exist | Create employee record first |
| "Insufficient leave" | Balance exhausted | Check remaining balance |
| "Date overlap" | Conflicting request | Check existing approved/pending requests |
| Slow embedding | Large file or slow CPU | Reduce file size, use GPU if available |

---

## License

Proprietary — All Rights Reserved.

---

**Version:** 4.1  
**Last Updated:** 2026-05-15  
**Python:** 3.12  
**AI Backend:** DeepSeek Chat / DeepSeek Reasoner  
**Frontend:** Vanilla JS + Chart.js  
**Built for:** 悦途汽车贸易有限公司
