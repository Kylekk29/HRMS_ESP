import json
import re
import time
import logging
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
import config

logger = logging.getLogger(__name__)


class AIModelProvider:
    """Wrapper around DeepSeek (OpenAI-compatible) via LangChain.
    
    增強功能:
    - 自動重試機制 (exponential backoff)
    - 請求超時處理
    - Token 限制自動截斷
    - 串流回應支援
    """

    # 最大輸入 token 估算 (DeepSeek 上下文約 128K，保守設 100K tokens)
    MAX_INPUT_CHARS = 180_000  # 約 45K tokens (中英文混合)
    MAX_OUTPUT_TOKENS = 16_384  # DeepSeek 最大輸出
    
    def __init__(self):
        logger.info("Initializing AI Model Provider...")
        logger.info(f"  Model: {config.MODEL_NORMAL}")
        logger.info(f"  Provider: {config.AI_PROVIDER}")
        logger.info(f"  Base URL: {config.BASE_URL}")
        
        self.llm = ChatOpenAI(
            model=config.MODEL_NORMAL,
            openai_api_key=config.API_KEY,
            openai_api_base=config.BASE_URL,
            temperature=0.1,
            timeout=config.TIMEOUT,
            max_retries=config.MAX_RETRIES,
            max_tokens=self.MAX_OUTPUT_TOKENS,
        )
        
        try:
            with open("prompts.json", "r", encoding="utf-8") as f:
                self.prompts = json.load(f)
            logger.info(f"Loaded prompts for {len(self.prompts)} features")
        except FileNotFoundError:
            logger.warning("prompts.json not found, using defaults")
    # ──────────────────────────────────────────────
    # 輸入截斷 (防止超過上下文限制)
    # ──────────────────────────────────────────────

    def _truncate_context(self, context: str, feature: str = "general") -> str:
        """智慧截斷上下文，保留開頭和結尾"""
        if len(context) <= self.MAX_INPUT_CHARS:
            return context
        
        logger.warning(
            f"Context too long ({len(context)} chars), truncating to {self.MAX_INPUT_CHARS} chars"
        )
        
        # 保留前 60% 和後 40%
        keep_head = int(self.MAX_INPUT_CHARS * 0.6)
        keep_tail = self.MAX_INPUT_CHARS - keep_head - 50  # 50 for separator
        
        head = context[:keep_head]
        tail = context[-keep_tail:] if keep_tail > 0 else ""
        
        truncated = (
            f"{head}\n\n... [Middle section truncated: "
            f"{len(context) - self.MAX_INPUT_CHARS} characters omitted] ...\n\n{tail}"
        )
        
        logger.info(f"  Truncated context: {len(context)} → {len(truncated)} chars")
        return truncated

    # ──────────────────────────────────────────────
    # JSON 提取
    # ──────────────────────────────────────────────

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from AI response with fallback strategies"""
        logger.info("Extracting JSON from AI response...")
        logger.info(f"  Response length: {len(text)} characters")
        
        # 策略 1: 直接解析
        cleaned = re.sub(r"^```(?:json)?\s*|```\s*$", "", text.strip(), flags=re.MULTILINE).strip()
        try:
            result = json.loads(cleaned)
            logger.info("  ✅ Successfully parsed JSON directly")
            return result
        except json.JSONDecodeError:
            pass
        
        # 策略 2: 正則提取最外層 JSON 物件
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                logger.info("  ✅ Extracted JSON via regex (object)")
                return result
            except json.JSONDecodeError:
                pass
        
        # 策略 3: 正則提取 JSON 陣列
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                logger.info("  ✅ Extracted JSON via regex (array)")
                return result
            except json.JSONDecodeError:
                pass
        
        # 策略 4: 嘗試修復常見錯誤 (多餘逗號、單引號)
        try:
            fixed = cleaned.replace("'", '"')
            fixed = re.sub(r",\s*}", "}", fixed)
            fixed = re.sub(r",\s*\]", "]", fixed)
            result = json.loads(fixed)
            logger.info("  ✅ Parsed JSON after fixing common errors")
            return result
        except json.JSONDecodeError:
            pass
        
        logger.warning("  ❌ Failed to extract valid JSON from response")
        logger.debug(f"  Raw response (first 500 chars): {text[:500]}")
        
        return {"error": "Failed to parse AI response", "raw": text[:800], "results": []}

    # ──────────────────────────────────────────────
    # 統一呼叫 (含重試邏輯)
    # ──────────────────────────────────────────────

    def cv_screening_ai(
        self, context: str, feature: str, max_retries: int = 3, **kwargs
    ) -> dict | str:
        """
        Unified AI call with automatic retry and backoff.
        
        Args:
            context: 主要上下文
            feature: 功能名稱 (對應 prompts.json)
            max_retries: 最大重試次數
            **kwargs: 傳遞給 prompt 模板的額外參數
        """
        logger.info("-" * 40)
        logger.info(f"🤖 AI CALL: {feature}")
        logger.info(f"  Context length: {len(context)} characters")
        
        prompt_cfg = self.prompts.get(feature)
        if not prompt_cfg:
            logger.error(f"Unknown feature: {feature}")
            return {"error": f"Unknown feature: {feature}", "results": []}

        # 截斷過長上下文
        context = self._truncate_context(context, feature)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_cfg["system_prompt"]),
            ("user", prompt_cfg["user_template"]),
        ])
        chain = prompt | self.llm | StrOutputParser()
        inputs = {"ctx": context, **kwargs}
        
        logger.info(f"  System prompt: {len(prompt_cfg['system_prompt'])} chars")
        logger.info(f"  User template: {len(prompt_cfg['user_template'])} chars")
        logger.info(f"  Extra inputs: {list(kwargs.keys())}")
        
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"  Attempt {attempt}/{max_retries}: Sending request to AI..."
                )
                start_time = time.time()
                
                response = chain.invoke(inputs)
                
                elapsed = time.time() - start_time
                logger.info(f"  ✅ Response received in {elapsed:.2f}s")
                logger.info(f"  Response length: {len(response)} chars")
                
                if feature == "general_chat":
                    return response
                
                result = self._extract_json(response)
                
                if "error" in result:
                    logger.warning(f"  ⚠️ JSON parse failed on attempt {attempt}")
                    if attempt < max_retries:
                        wait = 2 ** attempt
                        logger.info(f"  Retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                else:
                    results_count = len(result.get("results", []))
                    logger.info(f"  ✅ Extracted {results_count} results")
                    return result
                    
            except (APIConnectionError, APITimeoutError) as e:
                last_error = e
                logger.warning(f"  ⚠️ Connection/timeout on attempt {attempt}: {e}")
                if attempt < max_retries:
                    wait = min(2 ** attempt, 30)  # cap at 30s
                    logger.info(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"  ❌ All {max_retries} attempts failed")
                    
            except RateLimitError as e:
                last_error = e
                logger.warning(f"  ⚠️ Rate limit hit on attempt {attempt}: {e}")
                if attempt < max_retries:
                    wait = min(5 * attempt, 60)  # 5s, 10s, 15s...
                    logger.info(f"  Waiting {wait}s for rate limit reset...")
                    time.sleep(wait)
                else:
                    logger.error(f"  ❌ All {max_retries} attempts failed due to rate limit")
                    
            except APIError as e:
                last_error = e
                logger.error(f"  ❌ API error: {e}")
                break  # 非網路錯誤，不重試
                
            except Exception as e:
                last_error = e
                logger.error(f"  ❌ Unexpected error on attempt {attempt}: {e}", exc_info=True)
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.info(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"  ❌ All attempts exhausted")

        return {
            "error": f"AI call failed after {max_retries} attempts: {last_error}",
            "results": [],
        }

    def chat(self, query: str, context: str = "", feature: str = None, **kwargs) -> str:
        """
        General purpose AI chat with raw text response (no JSON parsing).
        
        Args:
            query: The user's question
            context: Optional context to include
            system_prompt: Optional custom system prompt
            **kwargs: Additional parameters for the prompt template
        
        Returns:
            Raw text response from the AI
        """
        logger.info("-" * 40)
        logger.info(f"💬 AI CHAT: {query[:50]}...")
        logger.info(f"  Context length: {len(context)} characters")
        
        
        prompt_cfg = self.prompts.get(feature)
        if not prompt_cfg:
            logger.error(f"Unknown feature: {feature}")
            return {"error": f"Unknown feature: {feature}", "results": []}

        
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_cfg["system_prompt"]),
            ("user", prompt_cfg["user_template"]),
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        inputs = {"query": query, "context": context, **kwargs}
        
        try:
            start_time = time.time()
            response = chain.invoke(inputs)
            elapsed = time.time() - start_time
            logger.info(f"✅ Response received in {elapsed:.2f}s")
            logger.info(f"  Response length: {len(response)} chars")
            return response
        except Exception as e:
            logger.error(f"❌ Chat failed: {e}", exc_info=True)
            return f"I encountered an error: {str(e)}. Please try again."
