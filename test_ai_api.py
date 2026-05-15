"""
test_ai_api.py — AI API Connection Test Script (Fixed)
──────────────────────────────────────────────────────
使用 LangChain 測試 DeepSeek API 連接和基本功能

使用方法:
  python test_ai_api.py
  python test_ai_api.py --verbose
  python test_ai_api.py --test-all
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 測試 LangChain 導入
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
    logger.info("✅ LangChain imports successful")
except ImportError as e:
    logger.error(f"❌ Failed to import LangChain: {e}")
    logger.error("Please install: pip install langchain langchain-openai openai")
    sys.exit(1)

# 嘗試加載環境變量
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ Loaded .env file")
except ImportError:
    logger.warning("⚠️ python-dotenv not installed, using system environment variables")


class AIAPITester:
    """AI API 連接測試器"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = []
        
        # 從環境變量讀取設置
        self.api_key = os.getenv("API_KEY", "")
        self.base_url = os.getenv("BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("MODEL_NORMAL", "deepseek-chat")
        
        if not self.api_key:
            logger.error("❌ API_KEY not found in environment variables or .env file!")
            sys.exit(1)
        
        # 隱藏 API key 的部分內容用於顯示
        if len(self.api_key) > 10:
            self.masked_key = self.api_key[:6] + "..." + self.api_key[-4:]
        else:
            self.masked_key = "***"
        
        logger.info(f"🔧 Configuration:")
        logger.info(f"   Base URL: {self.base_url}")
        logger.info(f"   Model: {self.model}")
        logger.info(f"   API Key: {self.masked_key}")
        
        # 初始化 LangChain LLM
        self.llm = None
        self._init_llm()
    
    def _init_llm(self) -> bool:
        """初始化 LangChain LLM 客戶端"""
        try:
            logger.info("Initializing LangChain ChatOpenAI client...")
            start_time = time.time()
            
            self.llm = ChatOpenAI(
                model=self.model,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                temperature=0.1,
                timeout=60,  # 60秒超時
                max_retries=2,
                max_tokens=2000,
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✅ LLM client initialized in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM client: {e}")
            self.llm = None
            return False
    
    def test_basic_connection(self) -> Dict[str, Any]:
        """測試基本 API 連接"""
        logger.info("\n" + "="*60)
        logger.info("📡 TEST 1: Basic Connection")
        logger.info("="*60)
        
        if not self.llm:
            return self._error_result("LLM not initialized")
        
        try:
            # 修復：使用簡單的 prompt，不含 JSON 大括號
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Reply with a simple greeting in one sentence."),
                ("user", "Say hello!")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            
            logger.info("Sending test request...")
            start_time = time.time()
            
            response = chain.invoke({})
            
            elapsed = time.time() - start_time
            
            if response:
                logger.info(f"✅ Response received in {elapsed:.2f}s")
                logger.info(f"   Response: {response[:150]}")
                
                return {
                    "test": "Basic Connection",
                    "status": "✅ PASS",
                    "elapsed_seconds": round(elapsed, 2),
                    "response_preview": response[:100]
                }
            else:
                return self._error_result("Empty response")
            
        except APIConnectionError as e:
            return self._error_result("Connection failed - check network/proxy", e)
        except APITimeoutError as e:
            return self._error_result("Request timeout", e)
        except RateLimitError as e:
            return self._error_result("Rate limit exceeded", e)
        except APIError as e:
            return self._error_result("API error", e)
        except Exception as e:
            return self._error_result("Unexpected error", e)
    
    def test_json_output(self) -> Dict[str, Any]:
        """測試 JSON 格式輸出"""
        logger.info("\n" + "="*60)
        logger.info("📋 TEST 2: JSON Output")
        logger.info("="*60)
        
        if not self.llm:
            return self._error_result("LLM not initialized")
        
        try:
            # 修復：將 JSON 模板放在 user message 中，避免 LangChain 變數衝突
            system_prompt = (
                "You are an HR professional. Analyze the candidate and return a valid JSON object. "
                "Do NOT include markdown code blocks. Output ONLY the raw JSON."
            )
            
            user_prompt = (
                "Candidate Profile:\n"
                "Name: John Doe\n"
                "Experience: 5 years Python development\n"
                "Skills: Python, Django, PostgreSQL, Docker\n"
                "Education: BS Computer Science\n\n"
                "Return a JSON object with these fields:\n"
                '- "overall_score": integer 0-100\n'
                '- "skills": array of strings\n'
                '- "recommendation": string ("Recommended" or "Not Recommended")\n'
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", user_prompt)
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            
            logger.info("Sending JSON output test...")
            start_time = time.time()
            
            response = chain.invoke({})
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Response received in {elapsed:.2f}s")
            
            # 嘗試解析 JSON
            try:
                cleaned = response.strip()
                # 移除可能的 markdown 標記
                if cleaned.startswith("```"):
                    lines = cleaned.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    cleaned = "\n".join(lines)
                cleaned = cleaned.strip()
                
                data = json.loads(cleaned)
                logger.info(f"✅ JSON parsed successfully:")
                logger.info(f"   Score: {data.get('overall_score')}")
                logger.info(f"   Skills: {data.get('skills')}")
                logger.info(f"   Recommendation: {data.get('recommendation')}")
                
                return {
                    "test": "JSON Output",
                    "status": "✅ PASS",
                    "elapsed_seconds": round(elapsed, 2),
                    "json_valid": True,
                    "output": data
                }
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ JSON parse failed: {e}")
                logger.info(f"   Raw response: {response[:300]}")
                return {
                    "test": "JSON Output",
                    "status": "⚠️ PARTIAL (connection OK, JSON parse failed)",
                    "elapsed_seconds": round(elapsed, 2),
                    "json_valid": False,
                    "raw_response": response[:300]
                }
            
        except Exception as e:
            return self._error_result("Test failed", e)
    
    def test_large_context(self) -> Dict[str, Any]:
        """測試較大上下文的處理"""
        logger.info("\n" + "="*60)
        logger.info("📚 TEST 3: Large Context Handling")
        logger.info("="*60)
        
        if not self.llm:
            return self._error_result("LLM not initialized")
        
        try:
            # 生成一個較大的上下文
            policy_template = (
                "{num}. {policy_name}: {policy_detail}\n"
            )
            
            policies = [
                ("Work Hours", "Standard working hours are 9:00 AM to 6:00 PM, Monday through Friday."),
                ("Leave Policy", "Employees are entitled to 14 days of annual leave per year."),
                ("Remote Work", "Employees may work remotely up to 2 days per week with manager approval."),
                ("Performance Review", "Annual performance reviews are conducted in December."),
                ("Training", "Company provides $2000 annual training budget per employee."),
                ("Health Insurance", "Comprehensive health insurance provided from day one."),
                ("Retirement", "401(k) with 4% company match after 6 months."),
                ("Overtime", "Overtime is compensated at 1.5x hourly rate beyond 40 hours/week."),
                ("Holidays", "10 paid public holidays per year."),
                ("Sick Leave", "5 paid sick days per year."),
            ]
            
            large_text = "Company Policy Document:\n\n"
            for i in range(10):  # 重複10次
                for j, (name, detail) in enumerate(policies):
                    large_text += policy_template.format(num=i*10+j+1, policy_name=name, policy_detail=detail)
            
            logger.info(f"Context length: {len(large_text)} characters")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an HR assistant. Answer based on the provided policy. Be concise and specific."),
                ("user", "Based on this policy:\n\n{context}\n\nWhat is the annual leave entitlement and training budget? Answer in one sentence.")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            
            logger.info("Sending large context request...")
            start_time = time.time()
            
            response = chain.invoke({"context": large_text})
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Response received in {elapsed:.2f}s")
            logger.info(f"   Response: {response[:200]}")
            
            return {
                "test": "Large Context",
                "status": "✅ PASS",
                "elapsed_seconds": round(elapsed, 2),
                "context_length": len(large_text),
                "response_preview": response[:150]
            }
            
        except APITimeoutError as e:
            return self._error_result("Timeout with large context", e)
        except Exception as e:
            return self._error_result("Test failed", e)
    
    def test_retry_and_error_handling(self) -> Dict[str, Any]:
        """測試重試機制和錯誤處理"""
        logger.info("\n" + "="*60)
        logger.info("🔄 TEST 4: Retry & Error Handling")
        logger.info("="*60)
        
        if not self.llm:
            return self._error_result("LLM not initialized")
        
        # 測試正常請求來驗證重試機制存在
        max_retries = 3
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"  Attempt {attempt}/{max_retries}...")
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant. Respond with exactly one word."),
                    ("user", "What is the capital of France? Reply with just the city name.")
                ])
                
                chain = prompt | self.llm | StrOutputParser()
                response = chain.invoke({})
                
                logger.info(f"  ✅ Request succeeded on attempt {attempt}")
                logger.info(f"     Response: {response.strip()}")
                
                return {
                    "test": "Retry & Error Handling",
                    "status": "✅ PASS",
                    "attempts_needed": attempt,
                    "max_retries": max_retries,
                    "response": response.strip()
                }
                
            except (APIConnectionError, APITimeoutError) as e:
                last_error = e
                logger.warning(f"  ⚠️ Attempt {attempt} failed: {type(e).__name__}")
                if attempt < max_retries:
                    wait_time = min(2 ** attempt, 10)
                    logger.info(f"     Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
            except Exception as e:
                last_error = e
                logger.error(f"  ❌ Unexpected error: {type(e).__name__}: {e}")
                break
        
        return self._error_result(f"All {max_retries} attempts failed", last_error)
    
    def test_cv_screening_simulation(self) -> Dict[str, Any]:
        """模擬真實的 CV 篩選場景"""
        logger.info("\n" + "="*60)
        logger.info("🎯 TEST 5: CV Screening Simulation")
        logger.info("="*60)
        
        if not self.llm:
            return self._error_result("LLM not initialized")
        
        try:
            jd = (
                "Senior Python Developer: "
                "5+ years experience, Django/Flask, PostgreSQL, Docker, AWS, "
                "team leadership experience preferred."
            )
            
            cv = (
                "John Smith\n"
                "Senior Software Engineer with 6 years experience\n"
                "Skills: Python, Django, PostgreSQL, Docker, AWS, Kubernetes\n"
                "Led a team of 4 developers on a cloud migration project\n"
                "Education: MS Computer Science, Stanford University"
            )
            
            system_prompt = (
                "You are a senior HR recruiter. Evaluate the candidate against the job description. "
                "Return a valid JSON object (no markdown) with these fields:\n"
                '- "match_score": integer 0-100\n'
                '- "strengths": array of strings\n'
                '- "weaknesses": array of strings\n'
                '- "verdict": "Highly Suitable" or "Suitable" or "Not Suitable"'
            )
            
            user_prompt = (
                "Job Description:\n{jd}\n\n"
                "Candidate Resume:\n{cv}\n\n"
                "Evaluate and return JSON."
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", user_prompt)
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            
            logger.info("Sending CV screening simulation...")
            start_time = time.time()
            
            response = chain.invoke({"jd": jd, "cv": cv})
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Response received in {elapsed:.2f}s")
            
            # 解析 JSON
            try:
                cleaned = response.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    cleaned = "\n".join(lines)
                cleaned = cleaned.strip()
                
                data = json.loads(cleaned)
                logger.info(f"✅ CV Screening JSON parsed:")
                logger.info(f"   Score: {data.get('match_score')}")
                logger.info(f"   Strengths: {len(data.get('strengths', []))} items")
                logger.info(f"   Weaknesses: {len(data.get('weaknesses', []))} items")
                logger.info(f"   Verdict: {data.get('verdict')}")
                
                return {
                    "test": "CV Screening Simulation",
                    "status": "✅ PASS",
                    "elapsed_seconds": round(elapsed, 2),
                    "json_valid": True,
                    "match_score": data.get('match_score'),
                    "verdict": data.get('verdict')
                }
            except json.JSONDecodeError:
                logger.warning("⚠️ JSON parse failed, but connection works")
                return {
                    "test": "CV Screening Simulation",
                    "status": "⚠️ PARTIAL (JSON parse failed)",
                    "elapsed_seconds": round(elapsed, 2),
                    "json_valid": False
                }
            
        except Exception as e:
            return self._error_result("Test failed", e)
    
    def _error_result(self, message: str, error: Exception = None) -> Dict[str, Any]:
        """生成錯誤結果"""
        error_msg = str(error) if error else message
        error_type = type(error).__name__ if error else "Unknown"
        
        logger.error(f"❌ {error_msg}")
        if error and self.verbose:
            import traceback
            traceback.print_exc()
        
        return {
            "status": "❌ FAIL",
            "error": error_msg,
            "error_type": error_type
        }
    
    def run_all_tests(self):
        """運行所有測試"""
        logger.info("\n" + "🚀" * 30)
        logger.info("STARTING AI API TESTS")
        logger.info("🚀" * 30)
        
        tests = [
            ("Basic Connection", self.test_basic_connection),
            ("JSON Output", self.test_json_output),
            ("Large Context", self.test_large_context),
            ("Retry & Error Handling", self.test_retry_and_error_handling),
            ("CV Screening Simulation", self.test_cv_screening_simulation),
        ]
        
        for name, test_func in tests:
            try:
                result = test_func()
                result["test"] = name
                self.results.append(result)
                time.sleep(0.5)  # 測試之間短暫休息
            except Exception as e:
                logger.error(f"Test '{name}' crashed: {e}")
                self.results.append({
                    "test": name,
                    "status": "💥 CRASHED",
                    "error": str(e)
                })
        
        self.print_summary()
    
    def print_summary(self):
        """打印測試摘要"""
        logger.info("\n" + "="*60)
        logger.info("📊 TEST SUMMARY")
        logger.info("="*60)
        
        if not self.results:
            logger.warning("No test results available")
            return
        
        passed = sum(1 for r in self.results if "PASS" in str(r.get("status", "")))
        failed = sum(1 for r in self.results if "FAIL" in str(r.get("status", "")))
        partial = sum(1 for r in self.results if "PARTIAL" in str(r.get("status", "")))
        crashed = sum(1 for r in self.results if "CRASHED" in str(r.get("status", "")))
        
        total = len(self.results)
        
        logger.info(f"\n   Total Tests: {total}")
        logger.info(f"   ✅ Passed:    {passed}")
        logger.info(f"   ⚠️  Partial:   {partial}")
        logger.info(f"   ❌ Failed:    {failed}")
        logger.info(f"   💥 Crashed:   {crashed}")
        
        # 詳細結果
        logger.info("\n📋 Detailed Results:")
        for i, result in enumerate(self.results, 1):
            test_name = result.get("test", f"Test {i}")
            status = result.get("status", "Unknown")
            elapsed = result.get("elapsed_seconds", "N/A")
            
            if "error" in result:
                logger.info(f"   {i}. {test_name}: {status}")
                logger.info(f"      Error: {result['error'][:150]}")
            elif "match_score" in result:
                logger.info(f"   {i}. {test_name}: {status} (Score: {result['match_score']}, {elapsed}s)")
            else:
                logger.info(f"   {i}. {test_name}: {status} ({elapsed}s)")
        
        # 評分
        score = (passed / total) * 100 if total > 0 else 0
        
        logger.info(f"\n🎯 Overall Score: {score:.0f}% ({passed}/{total} passed)")
        
        # 建議
        logger.info("\n💡 Recommendations:")
        if failed > 0 or crashed > 0:
            logger.info("   1. Check your API key and network connection")
            logger.info("   2. Verify the API endpoint is accessible")
            logger.info("   3. Check if behind a corporate firewall/proxy")
            logger.info("   4. Try with a different network or VPN")
        
        if partial > 0:
            logger.info("   → Some tests returned responses but JSON parsing failed")
            logger.info("   → This usually indicates the model is working but prompt needs tuning")
        
        if score == 100:
            logger.info("   🎉 All tests passed! API is working perfectly.")
        elif score >= 80:
            logger.info("   👍 Most tests passed. API is functional.")
        
        logger.info("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Test AI API connection using LangChain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_ai_api.py                 # Default: run first 2 tests
  python test_ai_api.py --quick         # Quick: basic connection only
  python test_ai_api.py --test-all      # Full: all 5 tests
  python test_ai_api.py -v --test-all   # Verbose: all tests with debug output
        """
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose/debug output")
    parser.add_argument("--test-all", "-a", action="store_true", help="Run all 5 tests")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick test only (basic connection)")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    tester = AIAPITester(verbose=args.verbose)
    
    if args.quick:
        result = tester.test_basic_connection()
        result["test"] = "Quick Connection Test"
        tester.results.append(result)
        tester.print_summary()
    elif args.test_all:
        tester.run_all_tests()
    else:
        # 默認：前兩個測試
        logger.info("Running default tests (basic + JSON output)...")
        result = tester.test_basic_connection()
        result["test"] = "Basic Connection"
        tester.results.append(result)
        time.sleep(0.5)
        result = tester.test_json_output()
        result["test"] = "JSON Output"
        tester.results.append(result)
        tester.print_summary()
    
    # 返回退出碼
    failed = sum(1 for r in tester.results if "FAIL" in str(r.get("status", "")))
    crashed = sum(1 for r in tester.results if "CRASHED" in str(r.get("status", "")))
    
    sys.exit(1 if (failed > 0 or crashed > 0) else 0)


if __name__ == "__main__":
    main()