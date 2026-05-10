"""
task_router.py  — AI-HR Bridge Platform v4.0
──────────────────────────────────────────────
Orchestrates HR workflows:
  1. Company-culture document upload & indexing
  2. Batch CV screening (embed all → single AI comparison call)
  3. Employee risk / termination analysis
  4. Interview transcript analysis (NEW)

v4.0 additions:
  - analyze_interview(): AI-powered interview evaluation
  - Screening history CRUD (unchanged)
"""
import logging
import os
import json
import time
from typing import List, Dict, Optional
from datetime import datetime

import config
from model_provider import AIModelProvider
from embedding_mgr import EmbeddingManager

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "core_competency": 30,
    "experience": 25,
    "education": 10,
    "culture_fit": 15,
    "development": 20,
}


class TaskRouter:

    def __init__(self):
        logger.info("=" * 60)
        logger.info("Initializing TaskRouter...")
        start_time = time.time()

        self.ai = AIModelProvider()
        self.emb = EmbeddingManager()
        self._culture_db_id: str | None = None

        os.makedirs(config.SCREENING_HISTORY_DIR, exist_ok=True)

        logger.info(f"TaskRouter initialized in {time.time() - start_time:.2f}s")
        logger.info("=" * 60)

    # ──────────────────────────────────────────────
    # 1. Company culture upload
    # ──────────────────────────────────────────────

    async def upload_company_culture(self, file_path: str, file_name: str) -> Dict:
        logger.info("=" * 60)
        logger.info(f"📄 UPLOADING COMPANY CULTURE: {file_name}")
        start_time = time.time()

        try:
            result = self.emb.embed_file_with_versioning(
                file_path, "company_culture", config.CULTURE_DB_DIR, "culture",
            )
            if "error" in result:
                logger.error(f"Failed: {result['error']}")
                return {"success": False, "message": result["error"], "db_id": None}

            self._culture_db_id = result["db_id"]
            elapsed = time.time() - start_time

            response = {
                "success": True, "message": result["message"],
                "db_id": result["db_id"], "action": result["action"],
                "version_number": result.get("version_number"),
                "chunk_count": result.get("chunk_count", 0),
                "is_new": result["is_new"],
            }
            logger.info(f"✅ Culture upload complete in {elapsed:.2f}s")
            return response
        except Exception as exc:
            logger.error(f"❌ Upload failed: {exc}", exc_info=True)
            return {"success": False, "message": str(exc), "db_id": None}

    # ──────────────────────────────────────────────
    # 2. Batch CV screening
    # ──────────────────────────────────────────────

    async def batch_screen_cvs(
        self,
        jd: str,
        cv_files: List[Dict],
        weights: Optional[Dict[str, int]] = None,
    ) -> Dict:
        logger.info("=" * 60)
        logger.info("🔍 BATCH CV SCREENING STARTED")
        logger.info(f"  Files: {len(cv_files)}, JD length: {len(jd)}")

        overall_start = time.time()

        if weights is None:
            weights = DEFAULT_WEIGHTS.copy()
        else:
            full_weights = DEFAULT_WEIGHTS.copy()
            full_weights.update(weights)
            weights = full_weights

        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] = round(weights[k] / total * 100)

        logger.info(f"  Weights: {weights}")

        if len(cv_files) > config.MAX_CV_BATCH_SIZE:
            cv_files = cv_files[:config.MAX_CV_BATCH_SIZE]

        errors: List[Dict] = []

        logger.info("STEP 1: Embedding CVs...")
        embed_results = self.emb.embed_cv_batch(cv_files)

        logger.info("STEP 2: Culture context...")
        culture_ctx = self._get_culture_context(jd)

        logger.info("STEP 3: Building contexts...")
        candidate_sections: List[str] = []
        successful_meta: List[Dict] = []

        for embed in embed_results:
            if "error" in embed:
                errors.append({"file": embed["file_name"], "error": embed["error"]})
                continue
            try:
                db = self.emb.load_db(embed["db_id"], config.CV_DB_DIR)
                docs = db.similarity_search(jd, k=config.CV_RETRIEVAL_K)
                cv_text = "\n".join(d.page_content for d in docs)
                section = (
                    f"=== CANDIDATE FILE: {embed['file_name']} ===\n"
                    f"{cv_text}\n"
                    f"=== END CANDIDATE: {embed['file_name']} ==="
                )
                candidate_sections.append(section)
                successful_meta.append({
                    "file_name": embed["file_name"],
                    "file_id": embed["file_id"],
                    "db_id": embed["db_id"],
                    "action": embed["action"],
                    "is_new": embed["is_new"],
                    "version_number": embed.get("version_number"),
                })
            except Exception as exc:
                errors.append({"file": embed.get("file_name", "?"), "error": str(exc)})

        if not candidate_sections:
            return {
                "success": False, "total_processed": len(cv_files),
                "successful": 0, "failed": len(errors),
                "screening_results": [], "errors": errors,
                "message": "All CV embeddings failed.",
            }

        logger.info("STEP 4: AI screening with weights...")
        all_candidates_ctx = "\n\n".join(candidate_sections)

        weights_text_full = (
            f"Core Competency (core_competency_match): {weights.get('core_competency', 30)}%\n"
            f"Experience (experience_match): {weights.get('experience', 25)}%\n"
            f"Education (education_match): {weights.get('education', 10)}%\n"
            f"Culture Fit (culture_fit_score): {weights.get('culture_fit', 15)}%\n"
            f"Development Potential (development_potential): {weights.get('development', 20)}%\n"
            f"\nOverall Score Formula: "
            f"overall_score = "
            f"(core_competency_match × {weights.get('core_competency', 30)} + "
            f"experience_match × {weights.get('experience', 25)} + "
            f"education_match × {weights.get('education', 10)} + "
            f"culture_fit_score × {weights.get('culture_fit', 15)} + "
            f"development_potential × {weights.get('development', 20)}) / 100"
        )

        ai_raw = self.ai.cv_screening_ai(
            all_candidates_ctx, "cv_screening",
            jd=jd, culture_ctx=culture_ctx,
            weights_text=weights_text_full,
        )

        if isinstance(ai_raw, dict) and "results" in ai_raw:
            results_list = ai_raw["results"]
        elif isinstance(ai_raw, list):
            results_list = ai_raw
        else:
            results_list = []

        for i, res in enumerate(results_list):
            if i < len(successful_meta):
                res.setdefault("candidate_file", successful_meta[i]["file_name"])
                res["_meta"] = {
                    "db_id": successful_meta[i]["db_id"],
                    "action": successful_meta[i]["action"],
                    "is_new": successful_meta[i]["is_new"],
                    "version": successful_meta[i].get("version_number"),
                }

        screening_record = {
            "screening_id": f"SCR_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(3).hex()}",
            "timestamp": datetime.now().isoformat(),
            "jd": jd[:2000],
            "weights": weights,
            "total_candidates": len(cv_files),
            "successful_candidates": len(successful_meta),
            "results": results_list,
            "ranking_summary": ai_raw.get("ranking_summary", "") if isinstance(ai_raw, dict) else "",
        }
        self._save_screening_history(screening_record)

        overall_elapsed = time.time() - overall_start

        response = {
            "success": len(errors) == 0,
            "total_processed": len(cv_files),
            "successful": len(candidate_sections),
            "failed": len(errors),
            "screening_id": screening_record["screening_id"],
            "weights_used": weights,
            "screening_results": [{
                "results": results_list,
                "culture_context_used": bool(culture_ctx and culture_ctx != "No company culture rules provided."),
            }],
            "errors": errors,
        }

        logger.info(f"✅ Screening complete in {overall_elapsed:.2f}s: {screening_record['screening_id']}")
        return response

    # ──────────────────────────────────────────────
    # Screening history CRUD
    # ──────────────────────────────────────────────

    def _save_screening_history(self, record: Dict) -> None:
        history = self._load_screening_history()
        history.insert(0, record)
        history = history[:100]
        with open(config.SCREENING_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def _load_screening_history(self) -> List[Dict]:
        if os.path.exists(config.SCREENING_HISTORY_FILE):
            with open(config.SCREENING_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def get_screening_history(self, limit: int = 20, offset: int = 0) -> Dict:
        history = self._load_screening_history()
        total = len(history)
        page = history[offset:offset + limit]
        summaries = []
        for h in page:
            summaries.append({
                "screening_id": h["screening_id"],
                "timestamp": h["timestamp"],
                "jd_preview": h.get("jd", "")[:200],
                "weights": h.get("weights", {}),
                "total_candidates": h.get("total_candidates", 0),
                "successful_candidates": h.get("successful_candidates", 0),
                "ranking_summary": h.get("ranking_summary", "")[:300],
            })
        return {"total": total, "offset": offset, "limit": limit, "screenings": summaries}

    def get_screening_detail(self, screening_id: str) -> Optional[Dict]:
        history = self._load_screening_history()
        for h in history:
            if h["screening_id"] == screening_id:
                return h
        return None

    # ──────────────────────────────────────────────
    # 3. Employee chat
    # ──────────────────────────────────────────────

    async def employee_chat(self, employee_id: str, query: str) -> Dict:
        """Enhanced chat with full RAG from all employee data sources."""
        logger.info(f"👤 EMPLOYEE CHAT: {employee_id}")
        
        try:
            # Start building context from all available sources
            context_parts = []
            
            # Load HRMS data
            from main_api import hrms as _hrms
            emp = _hrms.get_employee(employee_id)
            if emp:
                context_parts.extend([
                    f"Name: {emp.get('full_name', 'N/A')}",
                    f"Position: {emp.get('position', 'N/A')}",
                    f"Department: {emp.get('department', 'N/A')}",
                    f"Status: {emp.get('status', 'N/A')}",
                    f"Hire Date: {emp.get('hire_date', 'N/A')}",
                    f"Notes: {emp.get('notes', 'N/A')}",
                ])
                
                # KPI data
                kpis = emp.get('kpi', [])
                if kpis:
                    context_parts.append("\nKPI History:")
                    for kpi in kpis[-5:]:  # Last 5 KPI entries
                        context_parts.append(
                            f"  {kpi.get('period', '')}: Score={kpi.get('score', 0)}, "
                            f"Rating={kpi.get('rating', '')}"
                        )
                
                # Leave balance
                leave = emp.get('leave', {})
                context_parts.append("\nLeave Balance:")
                for lt in ['annual_leave', 'sick_leave', 'personal_leave']:
                    total = leave.get(f'{lt}_total', 0)
                    used = leave.get(f'{lt}_used', 0)
                    context_parts.append(f"  {lt}: {used}/{total} remaining")
            
            # Load from vector DB (CV/Profile)
            rag_used = False
            for doc_type in ["cv", "profile"]:
                try:
                    db = self.emb.load_employee_db(employee_id, doc_type)
                    docs = db.similarity_search(query, k=3)
                    if docs:
                        rag_used = True
                        context_parts.append(f"\n=== {doc_type.upper()} Document Excerpts ===")
                        for i, doc in enumerate(docs):
                            # Limit each excerpt to 500 chars
                            excerpt = doc.page_content[:500]
                            context_parts.append(f"[Excerpt {i+1}]: {excerpt}")
                except FileNotFoundError:
                    logger.info(f"  No {doc_type} DB for {employee_id}")
                except Exception as exc:
                    logger.warning(f"  Error loading {doc_type}: {exc}")
            
            if not context_parts:
                return {
                    "success": False, 
                    "error": f"No data available for {employee_id}", 
                    "response": None
                }
            
            context = "\n".join(context_parts)
            rag_status = "RAG enhanced" if rag_used else "HRMS only"
            logger.info(f"  Context built ({rag_status}): {len(context)} chars")
            # Use the chat method for raw text response
            response = self.ai.chat(
                query=query,
                context=context,
                feature="employee_chat",
            )
            
            return {
                "success": True, 
                "employee_id": employee_id, 
                "response": response,
                "rag_used": rag_used,
                "error": None
            }
            
        except Exception as exc:
            logger.error(f"Employee chat failed: {exc}", exc_info=True)
            return {"success": False, "error": str(exc), "response": None}
    # ──────────────────────────────────────────────
    # 4. Interview analysis (NEW — Module 4)
    # ──────────────────────────────────────────────

    async def analyze_interview(
        self,
        transcript: str,
        jd: str,
        competency: str = "",
    ) -> Dict:
        """
        Evaluate an interview transcript using AI.
        Returns dimension scores, strengths, weaknesses, red flags, and hiring recommendation.
        """
        logger.info("🎤 INTERVIEW ANALYSIS")
        logger.info(f"  Transcript length: {len(transcript)} chars")
        logger.info(f"  JD length: {len(jd)} chars")

        if not transcript.strip():
            return {"success": False, "error": "Interview transcript cannot be empty.", "analysis": None}
        if not jd.strip():
            return {"success": False, "error": "Job description cannot be empty.", "analysis": None}

        try:
            result = self.ai.cv_screening_ai(
                transcript,
                "interview_assistant",
                jd=jd,
                competency=competency or "Not specified",
            )

            if isinstance(result, dict) and "error" not in result:
                return {"success": True, "analysis": result, "error": None}
            else:
                return {"success": False, "analysis": result, "error": result.get("error", "AI analysis failed")}

        except Exception as exc:
            logger.error(f"  ❌ Interview analysis failed: {exc}", exc_info=True)
            return {"success": False, "error": str(exc), "analysis": None}

    # ──────────────────────────────────────────────
    # Helper: culture context
    # ──────────────────────────────────────────────

    def _get_culture_context(self, query: str) -> str:
        if not self._culture_db_id:
            db_id = self.emb.version_mgr.get_current_db_id("company_culture")
            if db_id:
                self._culture_db_id = db_id
        if not self._culture_db_id:
            return "No company culture rules provided."
        try:
            db = self.emb.load_db(self._culture_db_id, config.CULTURE_DB_DIR)
            docs = db.similarity_search(query, k=config.CULTURE_RETRIEVAL_K)
            return "\n".join(d.page_content for d in docs)
        except Exception:
            return "No company culture rules provided."
