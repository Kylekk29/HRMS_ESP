import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import config

class AIModelProvider:
    def __init__(self):
        self.llm = ChatOpenAI(
            model='deepseek-chat',
            openai_api_key=config.API_KEY,
            openai_api_base=config.BASE_URL,
            temperature=0.1
        )
        with open("prompts.json", "r", encoding="utf-8") as f:
            self.prompts = json.load(f)

    def _extract_json(self, text: str):
        """Robustly extract JSON from AI response tags"""
        try:
            # Match content between ```json and ``` or just the first { and last }
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return json.loads(text)
        except:
            return {"results": [], "error": "Failed to parse AI response"}

    def ask_ai(self, context: str, feature: str, **kwargs):
        prompt_cfg = self.prompts.get(feature)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_cfg["system_prompt"]),
            ("user", prompt_cfg["user_template"])
        ])

        # For non-chat tasks, we use a string parser first then manually parse JSON
        # This is more stable for models that ignore "JSON-only" instructions.
        parser = StrOutputParser()
        chain = prompt | self.llm | parser
        
        inputs = {"ctx": context, **kwargs}

        try:
            response = chain.invoke(inputs)
            if feature == "general_chat":
                return response
            return self._extract_json(response)
        except Exception as e:
            print(f"AI Error: {e}")
            return {"error": str(e), "results": []}