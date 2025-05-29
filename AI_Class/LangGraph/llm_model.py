from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# .env 파일에서 환경 변수 로드
load_dotenv()

# Google Gemini 설정
def llm_model():
    """LLM 모델을 설정하는 함수"""
    gemini = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7)
    return gemini