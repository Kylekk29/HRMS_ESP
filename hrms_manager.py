"""
hrms_manager.py  — AI-HR Bridge Platform
─────────────────────────────────────────
HRMS data layer (v4.0)

New in v4.0:
  - Module 1: Attendance management (check-in/out, monthly summary, absences)
  - Module 2: Leave request workflow (submit / approve / reject / pending list)
"""

import json
import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
import config
import logging
logger = logging.getLogger(__name__)

# ── Default leave entitlements (Taiwan Labour Standards Act) ──
DEFAULT_LEAVE = {
    "annual_leave_total": 3,
    "annual_leave_used": 0,
    "sick_leave_total": 30,
    "sick_leave_used": 0,
    "personal_leave_total": 14,
    "personal_leave_used": 0,
    "maternity_leave_total": 0,
    "maternity_leave_used": 0,
    "special_leave_total": 0,
    "special_leave_used": 0,
}

DEFAULT_SALARY = {
    "base": 0,
    "currency": "HKD",
    "pay_cycle": "Monthly",
    "last_review": "",
    "bonus": 0,
}

# ── File paths ──
ATTENDANCE_FILE = os.path.join(config.HRMS_DATA_DIR, "attendance_records.json")
LEAVE_REQUESTS_FILE = os.path.join(config.HRMS_DATA_DIR, "leave_requests.json")


def _now() -> str:
    return datetime.now().isoformat()


def _today() -> str:
    return date.today().isoformat()


def _annual_leave_by_tenure(hire_date_str: str) -> int:
    """Calculate annual leave days based on Taiwan labour law tenure."""
    if not hire_date_str:
        return 3
    try:
        hire = date.fromisoformat(hire_date_str)
        months = (date.today() - hire).days // 30
        if months < 6:
            return 3
        if months < 12:
            return 3
        years = months // 12
        if years < 1:
            return 7
        if years < 2:
            return 10
        if years < 3:
            return 14
        if years < 5:
            return 14
        if years < 10:
            return 15
        return min(15 + (years - 10), 30)
    except (ValueError, TypeError):
        return 7


def _daterange(start: str, end: str) -> List[str]:
    """Return list of dates (YYYY-MM-DD) from start to end inclusive."""
    try:
        s = date.fromisoformat(start)
        e = date.fromisoformat(end)
        return [(s + timedelta(days=i)).isoformat() for i in range((e - s).days + 1)]
    except ValueError:
        return []


class HRMSManager:

    def __init__(self):
        self._path = config.HRMS_RECORDS_FILE
        self._data: Dict[str, Dict] = self._load()

    # ──────────────────────────────────────────────
    # Persistence — employee records
    # ──────────────────────────────────────────────

    def _load(self) -> Dict[str, Dict]:
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    # ──────────────────────────────────────────────
    # Persistence — attendance
    # ──────────────────────────────────────────────

    def _load_attendance(self) -> Dict:
        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"records": []}

    def _save_attendance(self, data: Dict) -> None:
        with open(ATTENDANCE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ──────────────────────────────────────────────
    # Persistence — leave requests
    # ──────────────────────────────────────────────

    def _load_leave_requests(self) -> Dict:
        if os.path.exists(LEAVE_REQUESTS_FILE):
            with open(LEAVE_REQUESTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"requests": []}

    def _save_leave_requests(self, data: Dict) -> None:
        with open(LEAVE_REQUESTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ──────────────────────────────────────────────
    # CRUD — employee records
    # ──────────────────────────────────────────────

    def create_employee(self, employee_id: str, payload: Dict) -> Dict:
        if employee_id in self._data:
            raise ValueError(f"Employee {employee_id} already exists. Use update instead.")

        hire_date = payload.get("hire_date", "")
        leave = {**DEFAULT_LEAVE, "annual_leave_total": _annual_leave_by_tenure(hire_date)}
        leave.update(payload.get("leave", {}))

        salary = {**DEFAULT_SALARY}
        salary.update(payload.get("salary", {}))

        record = {
            "employee_id": employee_id,
            "full_name": payload.get("full_name", ""),
            "email": payload.get("email", ""),
            "phone": payload.get("phone", ""),
            "department": payload.get("department", ""),
            "position": payload.get("position", ""),
            "employment_type": payload.get("employment_type", "Full-time"),
            "hire_date": hire_date,
            "status": payload.get("status", "Active"),
            "salary": salary,
            "leave": leave,
            "kpi": payload.get("kpi", []),
            "emergency_contact": payload.get("emergency_contact", {"name": "", "relationship": "", "phone": ""}),
            "notes": payload.get("notes", ""),
            "created_at": _now(),
            "updated_at": _now(),
        }

        self._data[employee_id] = record
        self._save()
        return record

    def get_employee(self, employee_id: str) -> Optional[Dict]:
        return self._data.get(employee_id)

    def list_employees(self, department: Optional[str] = None, status: Optional[str] = None) -> List[Dict]:
        employees = list(self._data.values())
        if department:
            employees = [e for e in employees if e.get("department", "").lower() == department.lower()]
        if status:
            employees = [e for e in employees if e.get("status", "").lower() == status.lower()]
        return employees

    def update_employee(self, employee_id: str, payload: Dict) -> Dict:
        if employee_id not in self._data:
            raise KeyError(f"Employee {employee_id} not found.")

        record = self._data[employee_id]

        for field in ["full_name", "email", "phone", "department", "position",
                      "employment_type", "hire_date", "status", "notes"]:
            if field in payload:
                record[field] = payload[field]

        if "salary" in payload:
            record["salary"].update(payload["salary"])
        if "leave" in payload:
            record["leave"].update(payload["leave"])
        if "emergency_contact" in payload:
            record["emergency_contact"].update(payload["emergency_contact"])
        if "kpi" in payload:
            record["kpi"] = payload["kpi"]

        record["updated_at"] = _now()
        self._save()
        return record

    def delete_employee(self, employee_id: str) -> bool:
        if employee_id not in self._data:
            return False
        del self._data[employee_id]
        self._save()
        return True

    # ──────────────────────────────────────────────
    # KPI helpers
    # ──────────────────────────────────────────────

    def add_kpi_entry(self, employee_id: str, kpi_entry: Dict) -> Dict:
        if employee_id not in self._data:
            raise KeyError(f"Employee {employee_id} not found.")
        self._data[employee_id]["kpi"].append({**kpi_entry, "added_at": _now()})
        self._data[employee_id]["updated_at"] = _now()
        self._save()
        return self._data[employee_id]

    # ──────────────────────────────────────────────
    # Leave balance helpers (direct deduction — legacy)
    # ──────────────────────────────────────────────

    def apply_leave(self, employee_id: str, leave_type: str, days: float) -> Dict:
        if employee_id not in self._data:
            raise KeyError(f"Employee {employee_id} not found.")

        leave = self._data[employee_id]["leave"]
        used_key = f"{leave_type}_used"
        total_key = f"{leave_type}_total"

        if used_key not in leave or total_key not in leave:
            raise ValueError(f"Unknown leave type: {leave_type}")

        remaining = leave[total_key] - leave[used_key]
        if days > remaining:
            raise ValueError(
                f"Insufficient {leave_type}: requested {days}, remaining {remaining}"
            )

        leave[used_key] = round(leave[used_key] + days, 1)
        self._data[employee_id]["updated_at"] = _now()
        self._save()
        return self._data[employee_id]["leave"]

    # ──────────────────────────────────────────────
    # Statistics
    # ──────────────────────────────────────────────

    def get_department_summary(self) -> Dict:
        summary: Dict[str, Any] = {}
        for emp in self._data.values():
            dept = emp.get("department") or "Unassigned"
            if dept not in summary:
                summary[dept] = {"headcount": 0, "active": 0, "total_salary": 0}
            summary[dept]["headcount"] += 1
            if emp.get("status") == "Active":
                summary[dept]["active"] += 1
            summary[dept]["total_salary"] += emp.get("salary", {}).get("base", 0)
        return summary

    def get_department_tree(self) -> List[Dict]:
        tree_root: Dict[str, Any] = {"name": "組織架構", "children": {}}

        for emp in self._data.values():
            dept_str = emp.get("department", "").strip()
            if not dept_str:
                dept_str = "未分配"

            parts = [p.strip() for p in dept_str.replace("/", ">").split(">") if p.strip()]
            if not parts:
                parts = ["未分配"]

            current_level = tree_root["children"]
            for part in parts:
                if part not in current_level:
                    current_level[part] = {
                        "name": part,
                        "headcount": 0,
                        "active": 0,
                        "children": {},
                        "employees": [],
                    }
                node = current_level[part]
                node["headcount"] += 1
                if emp.get("status") == "Active":
                    node["active"] += 1
                node["employees"].append({
                    "employee_id": emp["employee_id"],
                    "full_name": emp.get("full_name", ""),
                    "position": emp.get("position", ""),
                    "status": emp.get("status", ""),
                })
                current_level = node["children"]

        def convert(node_dict: Dict) -> List[Dict]:
            result = []
            for name, node in node_dict.items():
                result.append({
                    "name": node["name"],
                    "headcount": node["headcount"],
                    "active": node["active"],
                    "employees": node["employees"],
                    "children": convert(node["children"]),
                })
            return result

        return convert(tree_root["children"])

    def set_department_structure(self, employee_id: str, department_path: str) -> Dict:
        if employee_id not in self._data:
            raise KeyError(f"Employee {employee_id} not found.")
        self._data[employee_id]["department"] = department_path
        self._data[employee_id]["updated_at"] = _now()
        self._save()
        return self._data[employee_id]

    # ══════════════════════════════════════════════
    # MODULE 1: Attendance Management
    # ══════════════════════════════════════════════

    def record_check_in(self, employee_id: str) -> Dict:
        """Record employee check-in. Raises if already checked in today."""
        if employee_id not in self._data:
            raise KeyError(f"Employee {employee_id} not found.")

        att = self._load_attendance()
        today = _today()
        now_time = datetime.now().strftime("%H:%M")

        # Check for duplicate check-in
        for rec in att["records"]:
            if rec["employee_id"] == employee_id and rec["date"] == today:
                raise ValueError(f"Employee {employee_id} has already checked in today ({rec['check_in']}).")

        # Auto-detect late (after 09:30)
        status = "Late" if now_time > "09:30" else "Present"

        # Generate ID
        existing_today = [r for r in att["records"] if r["date"] == today]
        seq = len(existing_today) + 1
        att_id = f"ATT{today.replace('-', '')}_{seq:03d}"

        record = {
            "id": att_id,
            "employee_id": employee_id,
            "date": today,
            "check_in": now_time,
            "check_out": None,
            "work_hours": 0,
            "status": status,
            "overtime_hours": 0,
            "notes": "",
        }
        att["records"].append(record)
        self._save_attendance(att)
        return record

    def record_check_out(self, employee_id: str) -> Dict:
        """Record employee check-out and compute work hours."""
        att = self._load_attendance()
        today = _today()
        now_time = datetime.now().strftime("%H:%M")

        # Find today's record
        target = None
        for rec in att["records"]:
            if rec["employee_id"] == employee_id and rec["date"] == today:
                target = rec
                break

        if not target:
            raise ValueError(f"No check-in record found for {employee_id} today.")
        if target.get("check_out"):
            raise ValueError(f"Employee {employee_id} has already checked out today.")

        # Calculate work hours
        fmt = "%H:%M"
        ci = datetime.strptime(target["check_in"], fmt)
        co = datetime.strptime(now_time, fmt)
        work_hours = round((co - ci).total_seconds() / 3600, 1)
        overtime = round(max(0, work_hours - 8), 1)

        target["check_out"] = now_time
        target["work_hours"] = work_hours
        target["overtime_hours"] = overtime

        if work_hours < 4:
            target["status"] = "Half-day"

        self._save_attendance(att)
        return target

    def get_monthly_attendance(self, employee_id: str, year: int, month: int) -> List[Dict]:
        """Return all attendance records for an employee in a given month."""
        att = self._load_attendance()
        prefix = f"{year:04d}-{month:02d}"
        return [
            r for r in att["records"]
            if r["employee_id"] == employee_id and r["date"].startswith(prefix)
        ]

    def get_daily_attendance_summary(self, date_str: str) -> Dict:
        """Return attendance summary for all employees on a given date."""
        att = self._load_attendance()
        total_employees = len(self._data)
        daily = [r for r in att["records"] if r["date"] == date_str]

        present = sum(1 for r in daily if r["status"] in ("Present",))
        late = sum(1 for r in daily if r["status"] == "Late")
        half_day = sum(1 for r in daily if r["status"] == "Half-day")
        absent = sum(1 for r in daily if r["status"] == "Absent")
        on_leave = sum(1 for r in daily if r["status"] == "On Leave")

        return {
            "date": date_str,
            "total": total_employees,
            "present": present,
            "late": late,
            "half_day": half_day,
            "absent": absent,
            "on_leave": on_leave,
            "not_recorded": total_employees - len(daily),
        }

    def mark_absent(self, employee_id: str, date_str: str, reason: str) -> Dict:
        """Mark an employee as absent on a specific date."""
        if employee_id not in self._data:
            raise KeyError(f"Employee {employee_id} not found.")

        att = self._load_attendance()

        # Check for existing record
        for rec in att["records"]:
            if rec["employee_id"] == employee_id and rec["date"] == date_str:
                rec["status"] = "Absent"
                rec["notes"] = reason
                self._save_attendance(att)
                return rec

        # Create new absent record
        existing = [r for r in att["records"] if r["date"] == date_str]
        seq = len(existing) + 1
        att_id = f"ATT{date_str.replace('-', '')}_{seq:03d}"

        record = {
            "id": att_id,
            "employee_id": employee_id,
            "date": date_str,
            "check_in": None,
            "check_out": None,
            "work_hours": 0,
            "status": "Absent",
            "overtime_hours": 0,
            "notes": reason,
        }
        att["records"].append(record)
        self._save_attendance(att)
        return record

    # ══════════════════════════════════════════════
    # MODULE 2: Leave Request Workflow
    # ══════════════════════════════════════════════

    def submit_leave_request(
        self,
        employee_id: str,
        leave_type: str,
        start_date: str,
        end_date: str,
        reason: str,
    ) -> Dict:
        """Submit a leave request. Now checks pending requests against balance."""
        if employee_id not in self._data:
            raise KeyError(f"Employee {employee_id} not found.")

        emp = self._data[employee_id]
        leave = emp.get("leave", {})
        used_key = f"{leave_type}_used"
        total_key = f"{leave_type}_total"

        if used_key not in leave:
            raise ValueError(f"Unknown leave type: {leave_type}")

        # Calculate days
        dates = _daterange(start_date, end_date)
        days = len(dates)
        if days <= 0:
            raise ValueError("end_date must be >= start_date.")

        # 修复：计算包含所有已提交且Pending状态的天数
        total_committed_days = leave.get(used_key, 0)
        
        # 检查所有Pending状态的请假请求
        lr_data = self._load_leave_requests()
        for req in lr_data["requests"]:
            if req["employee_id"] == employee_id and req["status"] == "Pending" and req["leave_type"] == leave_type:
                pending_dates = _daterange(req["start_date"], req["end_date"])
                # 只计算没有重叠的天数
                new_dates = [d for d in pending_dates if d not in dates]
                overlapping_dates = len(pending_dates) - len(new_dates)
                total_committed_days += len(new_dates)

        # 检查余额（包括已使用的和即将使用的）
        remaining = leave.get(total_key, 0) - total_committed_days - days
        if leave_type != "sick_leave" and remaining < 0:
            raise ValueError(
                f"Insufficient {leave_type}: requested {days} days, "
                f"available {leave.get(total_key, 0) - leave.get(used_key, 0)} days "
                f"(additional {total_committed_days - leave.get(used_key, 0)} days pending)"
            )

        # Check date overlap with existing approved/pending requests
        for req in lr_data["requests"]:
            if req["employee_id"] == employee_id and req["status"] in ("Pending", "Approved"):
                overlap = set(_daterange(req["start_date"], req["end_date"])) & set(dates)
                if overlap:
                    raise ValueError(
                        f"Date overlap with existing request {req['request_id']} "
                        f"({req['start_date']} ~ {req['end_date']})."
                    )

        # Generate request_id
        today_compact = date.today().strftime("%Y%m%d")
        existing_today = [r for r in lr_data["requests"] if r["request_id"].startswith(f"LR{today_compact}")]
        seq = len(existing_today) + 1
        request_id = f"LR{today_compact}_{seq:03d}"

        request = {
            "request_id": request_id,
            "employee_id": employee_id,
            "employee_name": emp.get("full_name", ""),
            "department": emp.get("department", ""),
            "leave_type": leave_type,
            "start_date": start_date,
            "end_date": end_date,
            "days": days,
            "reason": reason,
            "status": "Pending",
            "submitted_at": _now(),
            "approved_by": None,
            "approved_at": None,
            "rejection_reason": None,
        }
        lr_data["requests"].append(request)
        self._save_leave_requests(lr_data)
        
        logger.info(f"Leave request submitted: {request_id}, {days} days of {leave_type} for {employee_id}")
        return request


    def approve_leave_request(self, request_id: str, approver_id: str) -> Dict:
        """Approve a leave request. Now handles attendance conflicts properly."""
        lr_data = self._load_leave_requests()
        req = next((r for r in lr_data["requests"] if r["request_id"] == request_id), None)
        if not req:
            raise KeyError(f"Leave request {request_id} not found.")
        if req["status"] != "Pending":
            raise ValueError(f"Request {request_id} is already {req['status']}.")

        # Deduct leave balance
        emp_id = req["employee_id"]
        self.apply_leave(emp_id, req["leave_type"], req["days"])

        # 修复：智能标记考勤，避免覆盖已有记录
        att = self._load_attendance()
        conflicts = []
        for d in _daterange(req["start_date"], req["end_date"]):
            existing = next(
                (r for r in att["records"] if r["employee_id"] == emp_id and r["date"] == d), None
            )
            if existing:
                # 如果有打卡记录，记录冲突但不覆盖
                if existing.get("check_in") or existing.get("check_out"):
                    logger.warning(f"⚠️ Attendance conflict on {d}: employee {emp_id} has clock records")
                    conflicts.append(d)
                    # 保留原始记录，添加备注
                    existing["notes"] = f"{existing.get('notes', '')} | Leave approved: {req['leave_type']}"
                else:
                    # 只有空记录才更新为On Leave
                    existing["status"] = "On Leave"
                    existing["notes"] = f"Approved leave: {req['leave_type']}"
            else:
                # 创建新的On Leave记录
                existing_day = [r for r in att["records"] if r["date"] == d]
                seq = len(existing_day) + 1
                att["records"].append({
                    "id": f"ATT{d.replace('-', '')}_{seq:03d}",
                    "employee_id": emp_id,
                    "date": d,
                    "check_in": None,
                    "check_out": None,
                    "work_hours": 0,
                    "status": "On Leave",
                    "overtime_hours": 0,
                    "notes": f"Approved leave: {req['leave_type']}",
                })
        
        self._save_attendance(att)

        req["status"] = "Approved"
        req["approved_by"] = approver_id
        req["approved_at"] = _now()
        if conflicts:
            req["conflicts_notes"] = f"Attendance conflicts on: {', '.join(conflicts)}"
        self._save_leave_requests(lr_data)
        
        logger.info(f"Leave request {request_id} approved" + 
                (f" with {len(conflicts)} attendance conflicts" if conflicts else ""))
        return req

    def reject_leave_request(self, request_id: str, approver_id: str, reason: str) -> Dict:
        """Reject a leave request."""
        lr_data = self._load_leave_requests()
        req = next((r for r in lr_data["requests"] if r["request_id"] == request_id), None)
        if not req:
            raise KeyError(f"Leave request {request_id} not found.")
        if req["status"] != "Pending":
            raise ValueError(f"Request {request_id} is already {req['status']}.")

        req["status"] = "Rejected"
        req["approved_by"] = approver_id
        req["approved_at"] = _now()
        req["rejection_reason"] = reason
        self._save_leave_requests(lr_data)
        return req

    def get_pending_leave_requests(self, department: Optional[str] = None) -> List[Dict]:
        """Return all pending leave requests, optionally filtered by department."""
        lr_data = self._load_leave_requests()
        pending = [r for r in lr_data["requests"] if r["status"] == "Pending"]
        if department:
            pending = [r for r in pending if r.get("department", "").lower() == department.lower()]
        return pending

    def get_all_leave_requests(self, employee_id: Optional[str] = None) -> List[Dict]:
        """Return all leave requests, optionally filtered by employee."""
        lr_data = self._load_leave_requests()
        reqs = lr_data["requests"]
        if employee_id:
            reqs = [r for r in reqs if r["employee_id"] == employee_id]
        return reqs

    def get_employee_leave_summary(self, employee_id: str) -> Dict:
        """Return leave balance summary for an employee."""
        if employee_id not in self._data:
            raise KeyError(f"Employee {employee_id} not found.")

        leave = self._data[employee_id].get("leave", {})
        leave_types = ["annual_leave", "sick_leave", "personal_leave", "maternity_leave", "special_leave"]
        summary = {}
        for lt in leave_types:
            total = leave.get(f"{lt}_total", 0)
            used = leave.get(f"{lt}_used", 0)
            summary[lt] = {
                "total": total,
                "used": used,
                "remaining": round(total - used, 1),
            }
        return summary
