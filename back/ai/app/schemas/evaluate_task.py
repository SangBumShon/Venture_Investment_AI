from pydantic import BaseModel, Field
from typing import Dict, Optional, Literal
from datetime import datetime

class EvaluationRequest(BaseModel):
    startup_name: str = Field(..., min_length=1, max_length=100, description="평가할 스타트업 이름")

class EvaluationResponse(BaseModel):
    task_id: str = Field(..., description="평가 작업 ID")
    message: str = Field(..., description="응답 메시지")
    status: Literal["started", "failed"] = Field(..., description="작업 시작 상태")

class StatusResponse(BaseModel):
    task_id: str = Field(..., description="평가 작업 ID")
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(..., description="현재 작업 상태")
    progress: int = Field(..., ge=0, le=100, description="진행률 (0-100)")
    current_step: str = Field(..., description="현재 진행 중인 단계")
    description: str = Field(..., description="현재 단계 설명")
    estimated_time: Optional[str] = Field(None, description="예상 완료 시간")
    error_message: Optional[str] = Field(None, description="오류 발생 시 오류 메시지")

class ResultResponse(BaseModel):
    task_id: str = Field(..., description="평가 작업 ID")
    startup_name: str = Field(..., description="평가된 스타트업 이름")
    status: Literal["completed", "failed"] = Field(..., description="평가 완료 상태")
    decision: Literal["투자", "보류"] = Field(..., description="최종 투자 판단")
    overall_score: int = Field(..., ge=0, le=100, description="종합 점수")
    scores: Dict[str, int] = Field(..., description="항목별 점수")
    analysis_details: Dict[str, str] = Field(..., description="항목별 상세 분석 내용")
    created_at: datetime = Field(..., description="평가 완료 시간")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="오류 코드")
    message: str = Field(..., description="오류 메시지")
    details: Optional[Dict] = Field(None, description="상세 오류 정보")
