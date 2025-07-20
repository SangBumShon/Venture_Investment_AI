# app/services/utils/config.py
import os
from pathlib import Path

class Config:
    """설정 관리 클래스"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
    NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # Reranker용
    
    # Paths
    BASE_DIR = Path.cwd()
    DATA_DIR = BASE_DIR / "data"
    FONT_DIR = BASE_DIR / "font"
    OUTPUT_DIR = BASE_DIR / "investment_reports"
    
    # Font paths
    NANUM_REG_TTF = FONT_DIR / "NanumGothic.ttf"
    NANUM_BOLD_TTF = FONT_DIR / "NanumGothicBold.ttf"
    
    # Pinecone settings
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "startup-analysis")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    
    # LLM settings
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 생성"""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.DATA_DIR.mkdir(exist_ok=True)