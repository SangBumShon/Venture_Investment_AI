# backend/app/models/schemas.py
from typing import TypedDict, Literal, Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field

class AgentState(TypedDict, total=False):
    startup_name: str
    상품_점수: int
    기술_점수: int
    성장률_점수: int
    시장성_점수: int
    경쟁사_점수: int
    최종_판단: Literal["투자", "보류"]
    보고서: str
    pdf_path: str
    상품_분석_근거: str
    기술_분석_근거: str
    성장률_분석_근거: str
    시장성_분석_근거: str
    경쟁사_분석_근거: str

class ChecklistItem(BaseModel):
    """체크리스트 항목"""
    question: str = Field(description="질문 내용")
    score: int = Field(description="점수 (0-10)", ge=0, le=10)
    reasoning: str = Field(description="판단 근거")
    sources: List[str] = Field(description="참고 자료 URL 또는 출처")

class AnalysisResult(BaseModel):
    """분석 결과를 위한 JSON 스키마"""
    items: List[Union[ChecklistItem, Dict[str, Any]]] = Field(description="각 체크리스트 항목별 분석 결과")
    total_score: int = Field(description="총점 (0-100)")
    summary: str = Field(description="종합 분석 요약")
    
    def get_total_score(self) -> int:
        """총점 반환 (items에서 계산하거나 total_score 사용)"""
        if hasattr(self, 'total_score') and self.total_score is not None:
            return self.total_score
        
        # items에서 점수 계산
        if self.items:
            total = 0
            count = 0
            for item in self.items:
                if isinstance(item, dict):
                    score = item.get('score', 0)
                else:
                    score = item.score
                total += score
                count += 1
            
            if count > 0:
                return min(100, int(total * 10 / count))  # 10점 만점을 100점으로 변환
        
        return 0

class StartupAnalysisRequest(BaseModel):
    startup_name: str = Field(description="분석할 스타트업 이름")

class StartupAnalysisResponse(BaseModel):
    startup_name: str
    final_decision: Literal["투자", "보류"]
    scores: Dict[str, int]
    report_path: str
    created_at: str