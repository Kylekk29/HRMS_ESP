"""
main_api.py  —  AI-HR Bridge Platform v4.0
────────────────────────────────────────────
FastAPI application exposing all HR platform endpoints.

AI Features:
  POST /api/upload_culture
  POST /api/screen_cvs
  POST /api/screen_cvs_with_weights
  POST /api/employee_chat
  POST /api/upload_employee
  POST /api/interview_assist              (NEW - Module 4)

HRMS (employee records):
  GET/POST   /api/hrms/employees
  GET/PUT/DELETE /api/hrms/employees/{id}
  POST       /api/hrms/employees/{id}/kpi
  POST       /api/hrms/employees/{id}/leave
  POST       /api/hrms/employees/{id}/edit_profile
  PUT        /api/hrms/employees/{id}/department

Attendance (Module 1):
  POST /api/attendance/checkin
  POST /api/attendance/checkout
  GET  /api/attendance/monthly/{id}
  GET  /api/attendance/daily
  POST /api/attendance/absent

Leave Management (Module 2):
  POST /api/leave/request
  POST /api/leave/approve/{request_id}
  POST /api/leave/reject/{request_id}
  GET  /api/leave/pending
  GET  /api/leave/all
  GET  /api/leave/summary/{employee_id}

Payroll (Module 3):
  GET  /api/payroll/{employee_id}
  GET  /api/payroll/{employee_id}/payslip
  GET  /api/payroll/department/{dept}
  GET  /api/payroll/{employee_id}/adjustment

Employee Development (Module 5):
  GET  /api/employees/{id}/skills
  GET  /api/employees/{id}/skill_gaps
  GET  /api/employees/{id}/course_recommendations
  POST /api/employees/{id}/development_plan

Internal Marketplace (Module 7):
  POST /api/marketplace/projects
  GET  /api/marketplace/projects
  GET  /api/marketplace/projects/{id}/matches
  POST /api/marketplace/projects/{id}/apply
  POST /api/marketplace/applications/{id}/approve
  POST /api/marketplace/applications/{id}/reject

System:
  GET  /api/dashboard
  GET  /api/health
  GET  /api/screening_history
  GET  /api/screening_history/{id}
  GET  /api/hrms/departments/summary
  GET  /api/hrms/departments/tree
"""

import json as _json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime, date, timedelta
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

import config
from embedding_mgr import EmbeddingManager
from hrms_manager import HRMSManager
from payroll_manager import PayrollManager
from development_manager import DevelopmentManager
from internal_marketplace import InternalMarketplace
from task_router import TaskRouter

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="AI-HR Bridge Platform", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ──
router = TaskRouter()
hrms = HRMSManager()
payroll = PayrollManager(hrms)
dev_mgr = DevelopmentManager(hrms, router.emb, ai_provider=router.ai)
marketplace = InternalMarketplace(hrms, dev_mgr)

_ALLOWED_EXT = {".pdf", ".txt", ".docx"}


def _validate_ext(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {_ALLOWED_EXT}")
    return ext


def _save_upload(file_content: bytes, filename: str) -> str:
    path = os.path.join(config.UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(file_content)
    return path


def _cleanup(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _ok(data=None, message: str = "OK") -> Dict:
    return {"success": True, "data": data, "message": message}


def _err(message: str, data=None) -> Dict:
    return {"success": False, "data": data, "message": message}


# ══════════════════════════════════════════════════════════
# Frontend
# ══════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(404, "index.html not found")


# ══════════════════════════════════════════════════════════
# AI: Company culture
# ══════════════════════════════════════════════════════════

@app.post("/api/upload_culture")
async def upload_culture(file: UploadFile = File(...)):
    _validate_ext(file.filename)
    content = await file.read()
    path = _save_upload(content, file.filename)
    try:
        result = await router.upload_company_culture(path, file.filename)
        return result
    finally:
        _cleanup(path)


# ══════════════════════════════════════════════════════════
# AI: Batch CV screening
# ══════════════════════════════════════════════════════════

@app.post("/api/screen_cvs")
async def screen_cvs(jd: str = Form(...), files: List[UploadFile] = File(...)):
    if not jd.strip():
        raise HTTPException(400, "Job description cannot be empty.")
    if not files:
        raise HTTPException(400, "Please upload at least one resume.")
    if len(files) > config.MAX_CV_BATCH_SIZE:
        raise HTTPException(400, f"Max {config.MAX_CV_BATCH_SIZE} resumes per batch.")

    cv_files, paths = [], []
    try:
        for f in files:
            _validate_ext(f.filename)
            content = await f.read()
            path = _save_upload(content, f.filename)
            paths.append(path)
            cv_files.append({"file_path": path, "file_name": f.filename})
        return await router.batch_screen_cvs(jd, cv_files)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Screening failed: {exc}")
    finally:
        for p in paths:
            _cleanup(p)


@app.post("/api/screen_cvs_with_weights")
async def screen_cvs_with_weights(
    jd: str = Form(...),
    files: List[UploadFile] = File(...),
    weights_json: str = Form("{}"),
):
    if not jd.strip():
        raise HTTPException(400, "Job description cannot be empty.")
    if not files:
        raise HTTPException(400, "Please upload at least one resume.")
    try:
        weights = _json.loads(weights_json) if weights_json else {}
    except _json.JSONDecodeError:
        raise HTTPException(400, "weights_json must be valid JSON.")

    cv_files, paths = [], []
    try:
        for f in files:
            _validate_ext(f.filename)
            content = await f.read()
            path = _save_upload(content, f.filename)
            paths.append(path)
            cv_files.append({"file_path": path, "file_name": f.filename})
        return await router.batch_screen_cvs(jd, cv_files, weights=weights)
    finally:
        for p in paths:
            _cleanup(p)


@app.get("/api/screening_history")
async def get_screening_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    return router.get_screening_history(limit=limit, offset=offset)


@app.get("/api/screening_history/{screening_id}")
async def get_screening_detail(screening_id: str):
    detail = router.get_screening_detail(screening_id)
    if not detail:
        raise HTTPException(404, "Screening record not found.")
    return detail

# ══════════════════════════════════════════════════════════
# AI: Employee Chat (NEW - RAG-based employee-specific AI chat)
# ══════════════════════════════════════════════════════════

@app.post("/api/employee_chat")
async def employee_chat(
    employee_id: str = Form(...),
    query: str = Form(...),
):
    if not employee_id.strip():
        raise HTTPException(400, "Employee ID cannot be empty.")
    if not query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    
    # Verify employee exists
    emp = hrms.get_employee(employee_id)
    if not emp:
        raise HTTPException(404, f"Employee {employee_id} not found.")
    
    try:
        # Gather all employee data for RAG context
        context_parts = []
        
        # 1. HRMS data
        context_parts.append(f"=== HRMS DATA ===")
        context_parts.append(f"Name: {emp.get('full_name', 'N/A')}")
        context_parts.append(f"Position: {emp.get('position', 'N/A')}")
        context_parts.append(f"Department: {emp.get('department', 'N/A')}")
        context_parts.append(f"Status: {emp.get('status', 'N/A')}")
        context_parts.append(f"Hire Date: {emp.get('hire_date', 'N/A')}")
        context_parts.append(f"Employment Type: {emp.get('employment_type', 'N/A')}")
        context_parts.append(f"Notes: {emp.get('notes', 'N/A')}")
        
        # 2. Salary info
        salary = emp.get('salary', {})
        context_parts.append(f"\n=== SALARY ===")
        context_parts.append(f"Base: {salary.get('base', 0)} {salary.get('currency', 'HKD')}")
        context_parts.append(f"Bonus: {salary.get('bonus', 0)}")
        
        # 3. Leave balance
        leave = emp.get('leave', {})
        context_parts.append(f"\n=== LEAVE BALANCE ===")
        for lt in ['annual_leave', 'sick_leave', 'personal_leave']:
            total = leave.get(f'{lt}_total', 0)
            used = leave.get(f'{lt}_used', 0)
            context_parts.append(f"{lt}: {used}/{total} (remaining: {total - used})")
        
        # 4. Emergency contact
        ec = emp.get('emergency_contact', {})
        if ec.get('name'):
            context_parts.append(f"\n=== EMERGENCY CONTACT ===")
            context_parts.append(f"Name: {ec.get('name', 'N/A')}")
            context_parts.append(f"Relationship: {ec.get('relationship', 'N/A')}")
            context_parts.append(f"Phone: {ec.get('phone', 'N/A')}")
        
        # 5. KPI data
        kpis = emp.get('kpi', [])
        if kpis:
            context_parts.append(f"\n=== KPI HISTORY ===")
            for kpi in kpis:
                context_parts.append(
                    f"Period: {kpi.get('period', 'N/A')}, "
                    f"Score: {kpi.get('score', 0)}, "
                    f"Rating: {kpi.get('rating', 'N/A')}, "
                    f"Comments: {kpi.get('comments', 'N/A')}"
                )
        
        # 6. Attendance (current month)
        now = datetime.now()
        attendance_records = hrms.get_monthly_attendance(employee_id, now.year, now.month)
        if attendance_records:
            context_parts.append(f"\n=== ATTENDANCE (Current Month) ===")
            for rec in attendance_records[-10:]:  # Last 10 records
                context_parts.append(
                    f"Date: {rec.get('date', 'N/A')}, "
                    f"Check-in: {rec.get('check_in', 'N/A')}, "
                    f"Check-out: {rec.get('check_out', 'N/A')}, "
                    f"Status: {rec.get('status', 'N/A')}"
                )
        
        # 7. Leave requests
        leave_requests = hrms.get_all_leave_requests(employee_id=employee_id)
        if leave_requests:
            context_parts.append(f"\n=== LEAVE REQUESTS ===")
            for lr in leave_requests[-5:]:  # Last 5 requests
                context_parts.append(
                    f"ID: {lr.get('request_id', 'N/A')}, "
                    f"Type: {lr.get('leave_type', 'N/A')}, "
                    f"Dates: {lr.get('start_date', 'N/A')} to {lr.get('end_date', 'N/A')}, "
                    f"Status: {lr.get('status', 'N/A')}"
                )
        
        # 8. Embedded documents (CV/Profile)
        if router.emb:
            for doc_type in ["cv", "profile"]:
                try:
                    db = router.emb.load_employee_db(employee_id, doc_type)
                    docs = db.similarity_search(query, k=3)
                    if docs:
                        context_parts.append(f"\n=== {doc_type.upper()} DOCUMENT (Relevant Excerpts) ===")
                        for i, doc in enumerate(docs):
                            context_parts.append(f"[Excerpt {i+1}]: {doc.page_content[:500]}")
                except FileNotFoundError:
                    context_parts.append(f"\n=== {doc_type.upper()} DOCUMENT: Not found ===")
                except Exception as exc:
                    context_parts.append(f"\n=== {doc_type.upper()} DOCUMENT: Error loading ({exc}) ===")
        
        
        combined_context = "\n".join(context_parts)
        
        # Call AI with RAG context
        result = router.ai.chat(
            query=query,
            context=combined_context,
            feature="employee_chat"
        )
        
        return {
            "success": True,
            "employee_id": employee_id,
            "query": query,
            "response": result if isinstance(result, str) else result.get("response", "No response generated"),
            "context_used": True,
        }
        
    except Exception as exc:
        logger.error(f"Employee chat failed for {employee_id}: {exc}", exc_info=True)
        raise HTTPException(500, f"Chat failed: {exc}")


# ══════════════════════════════════════════════════════════
# AI: Upload employee profile document
# ══════════════════════════════════════════════════════════

@app.post("/api/upload_employee")
async def upload_employee(
    employee_id: str = Form(...),
    doc_type: str = Form("profile"),
    file: UploadFile = File(...),
):
    if not employee_id.strip():
        raise HTTPException(400, "Employee ID cannot be empty.")
    _validate_ext(file.filename)

    # FIX #6: Validate the employee exists before embedding
    if not hrms.get_employee(employee_id):
        raise HTTPException(
            404,
            f"Employee '{employee_id}' not found in HRMS. "
            "Please create the employee record first.",
        )

    content = await file.read()
    path = _save_upload(content, file.filename)
    try:
        result = router.emb.embed_employee_document(path, employee_id, doc_type)
        if "error" in result:
            raise HTTPException(500, result["error"])

        # Module 6: Auto-classify document
        classification = None
        try:
            from langchain_community.document_loaders import (
                PyPDFLoader, TextLoader, Docx2txtLoader,
            )
            ext = os.path.splitext(file.filename)[1].lower()
            loaders = {
                ".pdf": PyPDFLoader,
                ".txt": lambda p: TextLoader(p, encoding="utf-8"),
                ".docx": Docx2txtLoader,
            }
            if ext in loaders:
                docs = loaders[ext](path).load()
                raw_text = " ".join(d.page_content for d in docs)[:1000]
                classification = router.ai.cv_screening_ai(
                    raw_text, "document_classifier",
                    filename=file.filename,
                )
        except Exception as cls_exc:
            classification = {"note": f"Classification skipped: {cls_exc}"}

        return {
            "success": True,
            "employee_id": employee_id,
            "doc_type": doc_type,
            "action": result["action"],
            "version_number": result.get("version_number"),
            "chunk_count": result.get("chunk_count", 0),
            "is_new": result["is_new"],
            "message": result["message"],
            "classification": classification,
        }
    finally:
        _cleanup(path)

# ══════════════════════════════════════════════════════════
# AI: Interview assistant (Module 4)
# ══════════════════════════════════════════════════════════

@app.post("/api/interview_assist")
async def interview_assist(
    transcript: str = Form(...),
    jd: str = Form(...),
    competency: str = Form(""),
):
    """Analyse an interview transcript and return AI evaluation scores."""
    if not transcript.strip():
        raise HTTPException(400, "Transcript cannot be empty.")
    if not jd.strip():
        raise HTTPException(400, "Job description cannot be empty.")
    try:
        result = await router.analyze_interview(transcript, jd, competency)
        return result
    except Exception as exc:
        raise HTTPException(500, f"Interview analysis failed: {exc}")


# ══════════════════════════════════════════════════════════
# HRMS: Employee records
# ══════════════════════════════════════════════════════════

@app.get("/api/hrms/employees")
async def list_employees(
    department: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
):
    return {"employees": hrms.list_employees(department=department, status=status)}


@app.post("/api/hrms/employees")
async def create_employee(payload: Dict[str, Any]):
    employee_id = payload.get("employee_id", "").strip()
    if not employee_id:
        raise HTTPException(400, "employee_id is required.")
    try:
        record = hrms.create_employee(employee_id, payload)
        return {"success": True, "employee": record}
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))
@app.get("/api/hrms/employees/{employee_id}")
async def get_employee(employee_id: str):
    record = hrms.get_employee(employee_id)
    if not record:
        raise HTTPException(404, f"Employee {employee_id} not found.")
    return record


@app.put("/api/hrms/employees/{employee_id}")
async def update_employee(employee_id: str, payload: Dict[str, Any]):
    try:
        record = hrms.update_employee(employee_id, payload)
        return {"success": True, "employee": record}
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.delete("/api/hrms/employees/{employee_id}")
async def delete_employee(employee_id: str):
    if not hrms.delete_employee(employee_id):
        raise HTTPException(404, f"Employee {employee_id} not found.")
    return {"success": True, "message": f"Employee {employee_id} deleted."}


@app.post("/api/hrms/employees/{employee_id}/kpi")
async def add_kpi(employee_id: str, payload: Dict[str, Any]):
    try:
        record = hrms.add_kpi_entry(employee_id, payload)
        return {"success": True, "kpi": record["kpi"]}
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/hrms/employees/{employee_id}/leave")
async def apply_leave(employee_id: str, payload: Dict[str, Any]):
    leave_type = payload.get("leave_type", "")
    days = payload.get("days", 0)
    if not leave_type:
        raise HTTPException(400, "leave_type is required.")
    if days <= 0:
        raise HTTPException(400, "days must be positive.")
    try:
        leave = hrms.apply_leave(employee_id, leave_type, days)
        return {"success": True, "leave": leave}
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/hrms/employees/{employee_id}/edit_profile")
async def edit_employee_profile(
    employee_id: str,
    content: str = Form(...),
    doc_type: str = Form("profile"),
):
    if not employee_id.strip():
        raise HTTPException(400, "Employee ID required")
    if not content.strip():
        raise HTTPException(400, "Profile content cannot be empty")

    # FIX #6: Validate the employee exists before embedding
    if not hrms.get_employee(employee_id):
        raise HTTPException(
            404,
            f"Employee '{employee_id}' not found in HRMS. "
            "Please create the employee record first.",
        )

    safe_id = EmbeddingManager.clean_id(employee_id)
    filename = f"{safe_id}_profile.txt"
    file_path = os.path.join(config.UPLOAD_DIR, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    try:
        result = router.emb.embed_employee_document(file_path, employee_id, doc_type)
        return {
            "success": True,
            "employee_id": employee_id,
            "doc_type": doc_type,
            "action": result.get("action"),
            "version_number": result.get("version_number"),
            "chunk_count": result.get("chunk_count", 0),
            "message": result.get("message", "Profile updated and re-embedded"),
            "is_new": result.get("is_new", True),
        }
    finally:
        # FIX #4: Always clean up the temporary file (was `finally: pass`)
        _cleanup(file_path)


# ══════════════════════════════════════════════════════════
# HRMS: Departments
# ══════════════════════════════════════════════════════════

@app.get("/api/hrms/departments/summary")
async def department_summary():
    return hrms.get_department_summary()


@app.get("/api/hrms/departments/tree")
async def department_tree():
    return {"departments": hrms.get_department_tree()}


@app.put("/api/hrms/employees/{employee_id}/department")
async def update_employee_department(employee_id: str, payload: Dict[str, Any]):
    department = payload.get("department", "").strip()
    if not department:
        raise HTTPException(400, "department is required.")
    try:
        record = hrms.set_department_structure(employee_id, department)
        return {"success": True, "employee": record}
    except KeyError as exc:
        raise HTTPException(404, str(exc))


# ══════════════════════════════════════════════════════════
# MODULE 1: Attendance Management
# ══════════════════════════════════════════════════════════

@app.post("/api/attendance/checkin")
async def check_in(payload: Dict[str, Any]):
    employee_id = payload.get("employee_id", "").strip()
    if not employee_id:
        raise HTTPException(400, "employee_id is required.")
    try:
        record = hrms.record_check_in(employee_id)
        return _ok(
            record,
            f"Check-in recorded for {employee_id} at "
            f"{record['check_in']} ({record['status']})",
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/attendance/checkout")
async def check_out(payload: Dict[str, Any]):
    employee_id = payload.get("employee_id", "").strip()
    if not employee_id:
        raise HTTPException(400, "employee_id is required.")
    try:
        record = hrms.record_check_out(employee_id)
        return _ok(record, f"Check-out recorded. Work hours: {record['work_hours']}h")
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/attendance/monthly/{employee_id}")
async def monthly_attendance(
    employee_id: str,
    year: int = Query(...),
    month: int = Query(...),
):
    records = hrms.get_monthly_attendance(employee_id, year, month)
    return _ok(records, f"{len(records)} records found for {year}-{month:02d}")


@app.get("/api/attendance/daily")
async def daily_summary(date_str: str = Query(..., alias="date")):
    summary = hrms.get_daily_attendance_summary(date_str)
    return _ok(summary)


@app.post("/api/attendance/absent")
async def mark_absent(payload: Dict[str, Any]):
    employee_id = payload.get("employee_id", "").strip()
    date_str = payload.get("date", "").strip()
    reason = payload.get("reason", "")
    if not employee_id or not date_str:
        raise HTTPException(400, "employee_id and date are required.")
    try:
        record = hrms.mark_absent(employee_id, date_str, reason)
        return _ok(record)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


# ══════════════════════════════════════════════════════════
# MODULE 2: Leave Management
# ══════════════════════════════════════════════════════════

@app.post("/api/leave/request")
async def submit_leave_request(payload: Dict[str, Any]):
    try:
        req = hrms.submit_leave_request(
            payload.get("employee_id", ""),
            payload.get("leave_type", ""),
            payload.get("start_date", ""),
            payload.get("end_date", ""),
            payload.get("reason", ""),
        )
        return _ok(req, f"Leave request {req['request_id']} submitted successfully.")
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/leave/approve/{request_id}")
async def approve_leave(request_id: str, payload: Dict[str, Any]):
    approver_id = payload.get("approver_id", "system")
    try:
        req = hrms.approve_leave_request(request_id, approver_id)
        return _ok(req, f"Leave request {request_id} approved.")
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/leave/reject/{request_id}")
async def reject_leave(request_id: str, payload: Dict[str, Any]):
    approver_id = payload.get("approver_id", "system")
    reason = payload.get("reason", "")
    try:
        req = hrms.reject_leave_request(request_id, approver_id, reason)
        return _ok(req, f"Leave request {request_id} rejected.")
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/leave/pending")
async def get_pending_leave(department: Optional[str] = Query(None)):
    requests = hrms.get_pending_leave_requests(department=department)
    return _ok(requests, f"{len(requests)} pending leave requests.")


@app.get("/api/leave/all")
async def get_all_leave(employee_id: Optional[str] = Query(None)):
    requests = hrms.get_all_leave_requests(employee_id=employee_id)
    return _ok(requests, f"{len(requests)} leave requests.")


@app.get("/api/leave/summary/{employee_id}")
async def leave_summary(employee_id: str):
    try:
        summary = hrms.get_employee_leave_summary(employee_id)
        return _ok(summary)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


# ══════════════════════════════════════════════════════════
# MODULE 3: Payroll
# ══════════════════════════════════════════════════════════

@app.get("/api/payroll/{employee_id}")
async def get_payroll(
    employee_id: str,
    year: int = Query(default=datetime.now().year),
    month: int = Query(default=datetime.now().month),
):
    try:
        result = payroll.calculate_monthly_salary(employee_id, year, month)
        return _ok(result)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/payroll/{employee_id}/payslip", response_class=PlainTextResponse)
async def get_payslip(
    employee_id: str,
    year: int = Query(default=datetime.now().year),
    month: int = Query(default=datetime.now().month),
):
    try:
        slip = payroll.generate_payslip(employee_id, year, month)
        return slip
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/payroll/department/{department}")
async def dept_payroll(
    department: str,
    year: int = Query(default=datetime.now().year),
    month: int = Query(default=datetime.now().month),
):
    try:
        result = payroll.calculate_department_payroll(department, year, month)
        return _ok(result)
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/payroll/{employee_id}/adjustment")
async def salary_adjustment(employee_id: str):
    try:
        result = payroll.suggest_annual_adjustment(employee_id)
        return _ok(result)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


# ══════════════════════════════════════════════════════════
# MODULE 5: Employee Development
# ══════════════════════════════════════════════════════════

@app.get("/api/employees/{employee_id}/skills")
async def get_skills(employee_id: str):
    try:
        skills = dev_mgr.extract_skills_from_employee(employee_id)
        return _ok(skills, f"{len(skills)} skills identified.")
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))



# ══════════════════════════════════════════════════════════
# MODULE 8: Enhanced Dashboard
# ══════════════════════════════════════════════════════════

@app.get("/api/dashboard")
async def dashboard():
    employees = hrms.list_employees()
    dept_summary = hrms.get_department_summary()
    recent_screenings = router.get_screening_history(limit=5, offset=0)

    now = datetime.now()
    this_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    new_this_month = [
        e for e in employees
        if e.get("created_at", "") >= this_month_start.isoformat()
    ]

    status_counts = {"Active": 0, "On Leave": 0, "Terminated": 0}
    for e in employees:
        s = e.get("status", "Active")
        if s in status_counts:
            status_counts[s] += 1

    recent_terminated = [
        e for e in employees
        if e.get("status") == "Terminated"
        and e.get("updated_at", "") >= (now.replace(day=1) - timedelta(days=30)).isoformat()
    ]

    # Attendance today
    today_str = date.today().isoformat()
    attendance_today = hrms.get_daily_attendance_summary(today_str)

    # Pending approvals
    pending_leave = len(hrms.get_pending_leave_requests())
    pending_apps = len(marketplace.get_pending_applications())

    # Payroll this month (quick estimate from base salaries)
    total_base = sum(
        e.get("salary", {}).get("base", 0)
        for e in employees
        if e.get("status") == "Active"
    )
    active_count = status_counts["Active"]

    # Active projects
    active_projects = marketplace.count_active_projects()

    return {
        "employee_stats": {
            "total": len(employees),
            "active": status_counts["Active"],
            "on_leave": status_counts["On Leave"],
            "terminated": status_counts["Terminated"],
            "new_this_month": len(new_this_month),
            "terminated_this_month": len(recent_terminated),
        },
        "department_count": len(dept_summary),
        "departments": dept_summary,
        "recent_screenings": recent_screenings,
        "turnover_rate_this_month": round(
            len(recent_terminated) / max(len(employees), 1) * 100, 1
        ),
        "attendance_today": attendance_today,
        "pending_approvals": {
            "leave_requests": pending_leave,
            "project_applications": pending_apps,
        },
        "payroll_this_month": {
            "total": total_base,
            "average": int(total_base / active_count) if active_count > 0 else 0,
        },
        "active_internal_projects": active_projects,
    }


# ══════════════════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "AI-HR Bridge Platform v4.0"}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI-HR Bridge Platform v4.0...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
