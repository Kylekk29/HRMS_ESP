"""
development_manager.py  — AI-HR Bridge Platform
─────────────────────────────────────────────────
Module 5: Employee Skill Extraction, Gap Analysis & Course Recommendations

Works with embedded employee documents (CV / profile) stored in FAISS.
Falls back gracefully if no documents are embedded yet.

FIX #7: extract_skills_from_employee now distinguishes between two failure modes:
  - FileNotFoundError  → expected absence (no doc embedded yet); log INFO and skip
  - Any other exception → unexpected error (e.g. corrupted index); log ERROR with
                          a clear alert so operators know the index needs repair,
                          and re-raise only when it would prevent any fallback
"""

import json
import logging
import os
import re
from datetime import date, datetime
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

COURSES_FILE = os.path.join(config.HRMS_DATA_DIR, "development_courses.json")
DEVELOPMENT_PLANS_FILE = os.path.join(config.HRMS_DATA_DIR, "development_plans.json")

# ── Skill keyword taxonomy ──
SKILL_KEYWORDS = [
    "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "C++", "C#",
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
    "Docker", "Kubernetes", "k8s", "Terraform", "Ansible", "Jenkins",
    "AWS", "Azure", "GCP", "Cloud",
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "NLP",
    "React", "Vue", "Angular", "Node.js", "FastAPI", "Django", "Flask",
    "PowerBI", "Tableau", "Excel", "VLOOKUP", "DAX",
    "Project Management", "PMP", "Agile", "Scrum", "Kanban",
    "Leadership", "Communication", "Teamwork", "Presentation",
    "HR", "Recruitment", "Payroll", "Labour Law",
    "Finance", "Accounting", "Budgeting", "Financial Analysis",
    "Sales", "Marketing", "SEO", "CRM", "Salesforce",
    "Mandarin", "English", "Japanese", "Spanish",
]

SKILL_PROFICIENCY_SIGNALS = {
    "advanced": ["expert", "senior", "lead", "principal", "advanced", "專家", "高級", "資深", "精通"],
    "intermediate": ["proficient", "experienced", "3 year", "3年", "4年", "5年", "熟練", "中級"],
    "beginner": ["familiar", "basic", "entry", "junior", "初級", "了解", "基礎"],
}




class DevelopmentManager:
    """Handles employee skill extraction, gap analysis, and course recommendations."""

    def __init__(self, hrms_manager, embedding_manager=None, ai_provider=None):
        logger.info("Initializing DevelopmentManager...")
        self.hrms = hrms_manager
        self.embedding_manager = embedding_manager  # 修复：统一命名
        self.ai = ai_provider 
        
        if self.embedding_manager is None:
            logger.warning(
                "⚠️  EmbeddingManager is None! Skill extraction will be "
                "limited to HRMS data only."
            )
            logger.warning(
                "   Features affected: skill extraction, gap analysis, "
                "course recommendations, payroll adjustments"
            )
        else:
            logger.info("✓ EmbeddingManager available for full skill extraction")

        os.makedirs(config.HRMS_DATA_DIR, exist_ok=True)
        logger.info("DevelopmentManager initialized.")

    # ──────────────────────────────────────────────
    # Course library
    # ──────────────────────────────────────────────

    def _load_courses(self) -> List[Dict]:
        with open(COURSES_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("courses", [])

    # ──────────────────────────────────────────────
    # Skill extraction
    # ──────────────────────────────────────────────

    def _extract_skills_from_text(self, text: str) -> List[Dict]:
        """Parse skill keywords from raw text using keyword matching."""
        found = []
        text_lower = text.lower()
        for skill in SKILL_KEYWORDS:
            if skill.lower() in text_lower:
                proficiency = "intermediate"
                idx = text_lower.find(skill.lower())
                window = text_lower[max(0, idx - 80): idx + 80]
                for level, signals in SKILL_PROFICIENCY_SIGNALS.items():
                    if any(s in window for s in signals):
                        proficiency = level
                        break
                found.append({"skill": skill, "proficiency": proficiency})
        # Deduplicate by skill
        seen = set()
        deduped = []
        for item in found:
            if item["skill"] not in seen:
                seen.add(item["skill"])
                deduped.append(item)
        return deduped

    
    def extract_skills_from_employee(self, employee_id: str) -> List[Dict]:
        logger.info(f"Extracting skills for employee: {employee_id}")
        emp = self.hrms.get_employee(employee_id)  # ✅ 正确的变量名
        if not emp:
            raise KeyError(f"Employee {employee_id} not found.")

        combined_text = " ".join([
            emp.get("position", ""),
            emp.get("notes", ""),
            emp.get("profile_document", ""),
            " ".join(k.get("comments", "") for k in emp.get("kpi", [])),
        ])
        # Try vector DB search
        vector_db_used = False
        if self.embedding_manager:
            for doc_type in ("cv", "profile"):
                try:
                    db = self.embedding_manager.load_employee_db(employee_id, doc_type)
                    docs = db.similarity_search(
                        "skills experience or certifications",
                        k=5,
                    )
                    vector_text = " ".join(d.page_content for d in docs)
                    combined_text += " " + vector_text
                    logger.info(
                        f"  ✓ Retrieved {len(docs)} chunks from {doc_type} vector DB"
                    )
                    vector_db_used = True
                    break

                except FileNotFoundError:
                    # FIX #7: Normal absence — no document uploaded yet for this type
                    logger.info(
                        f"  ℹ️  No {doc_type} database found for employee "
                        f"{employee_id} (document not yet embedded)"
                    )

                except Exception as exc:
                    logger.error(
                        f"  ❌ UNEXPECTED error loading {doc_type} DB for "
                        f"{employee_id}: {exc}  "
                        f"— This may indicate a corrupted FAISS index. "
                        f"ACTION REQUIRED: re-upload the employee document to "
                        f"rebuild the index.",
                        exc_info=True,
                    )
        else:
            logger.warning(
                f"  ⚠️ EmbeddingManager unavailable, using only HRMS text data "
                f"for {employee_id}"
            )

        if not vector_db_used:
            logger.info(
                f"  ℹ️ Skill extraction for {employee_id} based only on HRMS "
                f"records (limited accuracy)"
            )

        skills = self._extract_skills_from_text(combined_text)
        logger.info(
            f"  Found {len(skills)} skills for {employee_id}"
            + (" (limited extraction — HRMS fallback)" if not vector_db_used else "")
        )
        return skills