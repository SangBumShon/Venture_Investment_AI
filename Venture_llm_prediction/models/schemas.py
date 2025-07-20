"""
Pydantic schemas and models for AI Startup Investment Evaluation Agent
"""

from typing import TypedDict, Literal
from langchain_core.pydantic_v1 import BaseModel, Field


# JSON Output Parser Models
class ProductAnalysis(BaseModel):
    총점: int = Field(description="상품/서비스 평가 총점 (0-100)")
    분석_근거: str = Field(description="각 항목별 점수와 판단 근거를 자세히 작성")

class TechnologyAnalysis(BaseModel):
    총점: int = Field(description="기술력 평가 총점 (0-100)")
    분석_근거: str = Field(description="기술력 평가 근거와 세부 분석")

class GrowthAnalysis(BaseModel):
    총점: int = Field(description="성장률 평가 총점 (0-100)")
    분석_근거: str = Field(description="성장률 평가 근거와 시장 분석")

class MarketAnalysis(BaseModel):
    총점: int = Field(description="시장성 평가 총점 (0-100)")
    분석_근거: str = Field(description="시장성 평가 근거와 시장 분석")

class CompetitorAnalysis(BaseModel):
    총점: int = Field(description="경쟁사 분석 총점 (0-100)")
    분석_근거: str = Field(description="경쟁사 분석 근거와 경쟁 환경 분석")

class FinalJudgement(BaseModel):
    최종_판단: Literal["투자", "보류"] = Field(description="최종 투자 판단")
    판단_근거: str = Field(description="최종 판단 근거")


class AgentState(TypedDict, total=False):
    """Agent state schema shared across all agents"""
    startup_name: str
    상품_점수: int
    기술_점수: int
    성장률_점수: int
    시장성_점수: int
    경쟁사_점수: int
    최종_판단: Literal["투자", "보류"]
    보고서: str
    pdf_path: str
    # Analysis reasoning
    상품_분석_근거: str
    기술_분석_근거: str
    성장률_분석_근거: str
    시장성_분석_근거: str
    경쟁사_분석_근거: str 