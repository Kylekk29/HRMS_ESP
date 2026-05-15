"""
payroll_manager.py  — AI-HR Bridge Platform
────────────────────────────────────────────
Module 3: Payroll Calculation Engine

Computes monthly salary based on:
  - Base salary
  - Attendance (absences, late arrivals, overtime)
  - Leave deductions (personal/annual leave deducted; sick leave not deducted)
  - Bonuses
  - Annual adjustment suggestions based on KPI + tenure
"""

import json
import logging
import os
from calendar import month, monthrange
from datetime import date, datetime
from typing import Dict, List, Optional

import config
from hrms_manager import HRMSManager

logger = logging.getLogger(__name__)

PAYROLL_FILE = os.path.join(config.PAYROLL_RECORDS_DIR, "payroll_records.json")

# Leave types deducted from salary (sick leave is NOT deducted)
DEDUCTIBLE_LEAVE_TYPES = {"annual_leave", "personal_leave", "special_leave"}


def _now() -> str:
    return datetime.now().isoformat()


class PayrollManager:
    """Handles monthly payroll calculation, payslip generation, and salary adjustment suggestions."""

    def __init__(self, hrms_manager: HRMSManager):
        logger.info("Initializing PayrollManager...")
        self.hrms = hrms_manager
        os.makedirs(config.PAYROLL_RECORDS_DIR, exist_ok=True)
        logger.info("PayrollManager initialized.")

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _load_payroll_records(self) -> Dict:
        if os.path.exists(PAYROLL_FILE):
            with open(PAYROLL_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"records": []}

    def _save_payroll_records(self, data: Dict) -> None:
        with open(PAYROLL_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _working_days_in_month(self, year: int, month: int) -> int:
        """Count weekdays (Mon-Fri) in a given month."""
        _, days_in_month = monthrange(year, month)
        count = 0
        for d in range(1, days_in_month + 1):
            if date(year, month, d).weekday() < 5:  # 0=Mon, 4=Fri
                count += 1
        return count

    def _get_attendance_stats(self, employee_id: str, year: int, month: int) -> Dict:
        """Aggregate attendance stats for a given month with proper mutual exclusion."""
        records = self.hrms.get_monthly_attendance(employee_id, year, month)

        absent_days = 0
        late_days = 0  # 改为统计天数，而不是次数
        overtime_hours = 0.0
        leave_days = 0
        work_days = 0
        half_days = 0

        for r in records:
            status = r.get("status", "")
            
            # 修复：确保状态互斥
            if status == "Absent":
                absent_days += 1
            elif status == "On Leave":
                leave_days += 1
            elif status == "Late":
                late_days += 1
                work_days += 1  # 迟到仍算作出勤，但记为迟到
            elif status == "Half-day":
                half_days += 1
                work_days += 0.5  # 半天出勤算0.5天
            elif status == "Present":
                work_days += 1
            # 忽略其他未知状态

            overtime_hours += r.get("overtime_hours", 0)

        logger.debug(f"Attendance stats for {employee_id} {year}-{month:02d}: "
                    f"work={work_days}, absent={absent_days}, late={late_days}, "
                    f"half={half_days}, leave={leave_days}, ot={overtime_hours}h")

        return {
            "work_days": work_days,
            "absent_days": absent_days,
            "late_days": late_days,  # 改名以明确含义
            "half_days": half_days,
            "overtime_hours": round(overtime_hours, 1),
            "leave_days_taken": leave_days,
        }


    def calculate_monthly_salary(self, employee_id: str, year: int, month: int) -> Dict:
        """
        Calculate full monthly salary breakdown for an employee.
        Fixed: proper mutual exclusion in attendance handling.
        """
        logger.info(f"Calculating salary for {employee_id} — {year:04d}-{month:02d}")

        emp = self.hrms.get_employee(employee_id)
        if not emp:
            raise KeyError(f"Employee {employee_id} not found.")

        base_salary = emp.get("salary", {}).get("base", 0)
        bonus = emp.get("salary", {}).get("bonus", 0)

        work_days_in_month = self._working_days_in_month(year, month)
        daily_rate = base_salary / work_days_in_month if work_days_in_month > 0 else 0
        hourly_rate = daily_rate / 8

        att = self._get_attendance_stats(employee_id, year, month)

        # 修复：缺勤扣除（只针对完全缺勤，迟到和半天已在出勤天数中体现）
        absent_deduction = round(att["absent_days"] * daily_rate, 0)
        
        # 修复：迟到罚款（固定金额或比例，避免重复扣款）
        # 迟到按小时工资的2倍罚款，而非全天工资的10%
        late_penalty = round(att["late_days"] * hourly_rate * 2, 0)
        
        # 修复：半天出勤扣除
        half_day_deduction = round(att["half_days"] * daily_rate * 0.5, 0)

        # 只扣除非病假的请假天数
        lr_data = self.hrms._load_leave_requests()
        period_prefix = f"{year:04d}-{month:02d}"
        deductible_leave_days = 0
        for req in lr_data.get("requests", []):
            if (
                req["employee_id"] == employee_id
                and req["status"] == "Approved"
                and req.get("leave_type") in DEDUCTIBLE_LEAVE_TYPES
                and (req.get("start_date", "").startswith(period_prefix)
                    or req.get("end_date", "").startswith(period_prefix))
            ):
                deductible_leave_days += req.get("days", 0)

        leave_deduction = round(deductible_leave_days * daily_rate, 0)
        overtime_pay = round(att["overtime_hours"] * hourly_rate * 1.5, 0)

        gross_salary = int(
            base_salary
            - absent_deduction
            - late_penalty
            - half_day_deduction
            - leave_deduction
            + overtime_pay
            + bonus
        )

        result = {
            "employee_id": employee_id,
            "employee_name": emp.get("full_name", ""),
            "department": emp.get("department", ""),
            "period": f"{year:04d}-{month:02d}",
            "breakdown": {
                "base_salary": base_salary,
                "absent_deduction": int(absent_deduction),
                "late_penalty": int(late_penalty),
                "half_day_deduction": int(half_day_deduction),
                "leave_deduction": int(leave_deduction),
                "overtime_pay": int(overtime_pay),
                "bonus": bonus,
                "gross_salary": gross_salary,
            },
            "attendance_details": att,
            "working_days_in_month": work_days_in_month,
            "daily_rate": round(daily_rate, 2),
            "calculated_at": _now(),
        }

        logger.info(f"  Gross salary: {gross_salary:,} | "
                f"Deductions: absent={absent_deduction}, late={late_penalty}, "
                f"half_day={half_day_deduction}, leave={leave_deduction}")
        return result


    def generate_payslip(self, employee_id: str, year: int, month: int) -> str:
        """Return a formatted payslip text string with updated deduction items."""
        logger.info(f"Generating payslip for {employee_id} — {year:04d}-{month:02d}")
        data = self.calculate_monthly_salary(employee_id, year, month)
        b = data["breakdown"]
        att = data["attendance_details"]
        currency = self.hrms.get_employee(employee_id).get("salary", {}).get("currency", "HKD")

        lines = [
            "=" * 52,
            f"  PAYSLIP — {data['period']}",
            "=" * 52,
            f"  Employee : {data['employee_name']} ({data['employee_id']})",
            f"  Dept.    : {data.get('department', 'N/A')}",
            f"  Currency : {currency}",
            "-" * 52,
            "  EARNINGS",
            f"    Base Salary               {b['base_salary']:>12,}",
            f"    Overtime Pay              {b['overtime_pay']:>12,}",
            f"    Bonus                     {b['bonus']:>12,}",
            "-" * 52,
            "  DEDUCTIONS",
            f"    Absent Deduction         -{b['absent_deduction']:>12,}",
            f"    Late Penalty             -{b['late_penalty']:>12,}",
            f"    Half-day Deduction       -{b['half_day_deduction']:>12,}",
            f"    Leave Deduction          -{b['leave_deduction']:>12,}",
            "=" * 52,
            f"  GROSS SALARY              {b['gross_salary']:>12,}",
            "=" * 52,
            "  ATTENDANCE SUMMARY",
            f"    Work Days Recorded        {att['work_days']:>12}",
            f"    Absent Days               {att['absent_days']:>12}",
            f"    Late Days                 {att['late_days']:>12}",
            f"    Half Days                 {att['half_days']:>12}",
            f"    Overtime Hours            {att['overtime_hours']:>12}",
            f"    Leave Days Taken          {att['leave_days_taken']:>12}",
            "=" * 52,
            f"  Generated: {data['calculated_at'][:19]}",
            "=" * 52,
        ]
        return "\n".join(lines)
    # ──────────────────────────────────────────────
    # Department payroll summary
    # ──────────────────────────────────────────────

    def calculate_department_payroll(self, department: str, year: int, month: int) -> Dict:
        """Return aggregated payroll data for all employees in a department."""
        logger.info(f"Calculating department payroll: {department} — {year:04d}-{month:02d}")
        employees = self.hrms.list_employees(department=department)

        if not employees:
            return {
                "department": department,
                "period": f"{year:04d}-{month:02d}",
                "employee_count": 0,
                "total_base": 0,
                "total_overtime": 0,
                "total_gross": 0,
                "avg_salary": 0,
                "employees": [],
            }

        total_base = 0
        total_overtime = 0
        total_gross = 0
        emp_summaries = []

        for emp in employees:
            try:
                calc = self.calculate_monthly_salary(emp["employee_id"], year, month)
                total_base += calc["breakdown"]["base_salary"]
                total_overtime += calc["breakdown"]["overtime_pay"]
                total_gross += calc["breakdown"]["gross_salary"]
                emp_summaries.append({
                    "employee_id": emp["employee_id"],
                    "name": emp.get("full_name", ""),
                    "gross_salary": calc["breakdown"]["gross_salary"],
                })
            except Exception as exc:
                logger.warning(f"  Skipping {emp['employee_id']}: {exc}")

        count = len(emp_summaries)
        return {
            "department": department,
            "period": f"{year:04d}-{month:02d}",
            "employee_count": count,
            "total_base": int(total_base),
            "total_overtime": int(total_overtime),
            "total_gross": int(total_gross),
            "avg_salary": int(total_gross / count) if count > 0 else 0,
            "employees": emp_summaries,
        }

    # ──────────────────────────────────────────────
    # Annual adjustment suggestion
    # ──────────────────────────────────────────────

    def suggest_annual_adjustment(self, employee_id: str) -> Dict:
        """
        Suggest salary adjustment based on KPI average and tenure.
        KPI ≥90: +8–12%  | 80–89: +5–8%  | 70–79: +3–5%  | <70: +0–2%
        Tenure ≥3yr: +1%  | ≥5yr: +2%
        """
        logger.info(f"Generating salary adjustment suggestion for {employee_id}")
        emp = self.hrms.get_employee(employee_id)
        if not emp:
            raise KeyError(f"Employee {employee_id} not found.")

        current_salary = emp.get("salary", {}).get("base", 0)
        kpi_entries = emp.get("kpi", [])
        hire_date_str = emp.get("hire_date", "")

        # KPI average
        kpi_scores = [k.get("score", 0) for k in kpi_entries if k.get("score") is not None]
        kpi_avg = round(sum(kpi_scores) / len(kpi_scores), 1) if kpi_scores else 0

        # Tenure in years
        try:
            hire = date.fromisoformat(hire_date_str)
            years = round((date.today() - hire).days / 365.25, 1)
        except (ValueError, TypeError):
            years = 0

        # Base range from KPI
        if kpi_avg >= 90:
            base_min, base_max = 8, 12
            kpi_reason = f"KPI {kpi_avg}分 (卓越表現)"
        elif kpi_avg >= 80:
            base_min, base_max = 5, 8
            kpi_reason = f"KPI {kpi_avg}分 (優秀表現)"
        elif kpi_avg >= 70:
            base_min, base_max = 3, 5
            kpi_reason = f"KPI {kpi_avg}分 (良好表現)"
        else:
            base_min, base_max = 0, 2
            kpi_reason = f"KPI {kpi_avg}分 (需要改善)"

        # Tenure bonus
        tenure_bonus = 0
        tenure_reason = ""
        if years >= 5:
            tenure_bonus = 2
            tenure_reason = f"；年資滿5年加成 +2%"
        elif years >= 3:
            tenure_bonus = 1
            tenure_reason = f"；年資滿3年加成 +1%"

        final_min = base_min + tenure_bonus
        final_max = base_max + tenure_bonus
        mid_pct = (final_min + final_max) / 2
        suggested = int(current_salary * (1 + mid_pct / 100))

        reason = f"{kpi_reason}{tenure_reason}；建議調薪幅度 {final_min}%–{final_max}%"

        return {
            "employee_id": employee_id,
            "employee_name": emp.get("full_name", ""),
            "current_salary": current_salary,
            "kpi_avg": kpi_avg,
            "kpi_entries_count": len(kpi_scores),
            "years": years,
            "range": {"min": final_min, "max": final_max},
            "suggested": suggested,
            "reason": reason,
        }

    # ──────────────────────────────────────────────
    # Total payroll for all employees (dashboard)
    # ──────────────────────────────────────────────

    def get_total_payroll_this_month(self) -> Dict:
        """Quick summary of total payroll for the current month."""
        now = date.today()
        employees = self.hrms.list_employees(status="Active")
        total = 0
        count = 0
        for emp in employees:
            try:
                calc = self.calculate_monthly_salary(emp["employee_id"], now.year, now.month)
                total += calc["breakdown"]["gross_salary"]
                count += 1
            except Exception:
                pass
        return {
            "total": total,
            "average": int(total / count) if count > 0 else 0,
            "employee_count": count,
        }