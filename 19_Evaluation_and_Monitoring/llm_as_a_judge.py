import os
import json
import logging
from typing import Optional
from dotenv import load_dotenv

# Google Generative AI imports
try:
    import google.generativeai as genai
except ImportError:
    print("Error: 'google-generativeai' not found.")
    genai = None

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

LEGAL_SURVEY_RUBRIC = """
You are an expert evaluator. Rate the quality of the following survey question on a scale of 1-5.
Criteria:
1. Clarity & Precision
2. Neutrality & Bias
3. Relevance
4. Completeness

Output MUST be JSON:
{
  "overall_score": int,
  "rationale": "summary",
  "detailed_feedback": ["bullets"],
  "recommended_action": "text"
}
"""

class LLMJudge:
    """
    Uses an LLM to evaluate agent outputs based on a rubric.
    """
    def __init__(self, model_name: str = 'gemini-2.5-flash'):
        if genai:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name)
            else:
                self.model = None
        else:
            self.model = None

    def evaluate_question(self, question: str) -> Optional[dict]:
        if not self.model:
            print("Model not initialized.")
            return None

        prompt = f"{LEGAL_SURVEY_RUBRIC}\n\nQUESTION: {question}"
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            return None

if __name__ == "__main__":
    print("--- LLM as a Judge (G-Eval) Demo ---")
    judge = LLMJudge()
    
    test_q = "How do you feel about IP laws in Switzerland?"
    print(f"Evaluating Question: '{test_q}'")
    
    result = judge.evaluate_question(test_q)
    if result:
        print(json.dumps(result, indent=2))
