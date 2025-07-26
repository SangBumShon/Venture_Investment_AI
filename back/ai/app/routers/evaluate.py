import os
import httpx
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from typing import Optional

from ..schemas.evaluate_task import (
    EvaluationRequest, EvaluationResponse, StatusResponse, 
    ResultResponse, ErrorResponse
)
from ..services.evaluation_service import (
    start_evaluation, get_evaluation_status, get_evaluation_result, get_pdf_path
)

load_dotenv()

router = APIRouter(prefix="/api/evaluate", tags=["Evaluation"])

@router.post("", response_model=EvaluationResponse)
async def create_evaluation(request: EvaluationRequest):
    """스타트업 투자 평가 시작"""
    try:
        task_id = await start_evaluation(request.startup_name)
        return EvaluationResponse(
            task_id=task_id,
            message="평가가 시작되었습니다.",
            status="started"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "EVALUATION_START_FAILED",
                "message": f"평가 시작 중 오류가 발생했습니다: {str(e)}"
            }
        )

@router.get("/{task_id}/status", response_model=StatusResponse)
async def get_status(task_id: str):
    """평가 진행상황 조회"""
    status = get_evaluation_status(task_id)
    
    if not status:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "TASK_NOT_FOUND",
                "message": "해당 작업을 찾을 수 없습니다."
            }
        )
    
    return StatusResponse(
        task_id=task_id,
        status=status.get("status", "pending"),
        progress=status.get("progress", 0),
        current_step=status.get("current_step", ""),
        description=status.get("description", ""),
        estimated_time=status.get("estimated_time"),
        error_message=status.get("error_message")
    )

@router.get("/{task_id}/result", response_model=ResultResponse)
async def get_result(task_id: str):
    """평가 결과 조회"""
    try:
        result = get_evaluation_result(task_id)
        
        if not result:
            # 작업이 아직 완료되지 않았거나 존재하지 않는 경우
            status = get_evaluation_status(task_id)
            if not status:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "TASK_NOT_FOUND",
                        "message": "해당 작업을 찾을 수 없습니다."
                    }
                )
            elif status.get("status") == "failed":
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "EVALUATION_FAILED",
                        "message": status.get("error_message", "평가 중 오류가 발생했습니다.")
                    }
                )
            else:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "EVALUATION_IN_PROGRESS",
                        "message": "평가가 아직 완료되지 않았습니다."
                    }
                )
        
        # 결과 데이터 검증
        required_fields = ["startup_name", "decision", "overall_score", "scores", "analysis_details", "created_at"]
        for field in required_fields:
            if field not in result:
                print(f"결과에 필수 필드가 없습니다: {field}, result: {result}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "INVALID_RESULT",
                        "message": f"결과에 필수 필드가 없습니다: {field}"
                    }
                )
        
        return ResultResponse(
            task_id=task_id,
            startup_name=result["startup_name"],
            status="completed",
            decision=result["decision"],
            overall_score=result["overall_score"],
            scores=result["scores"],
            analysis_details=result["analysis_details"],
            created_at=result["created_at"]
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"결과 조회 중 예외 발생: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": f"결과 조회 중 오류가 발생했습니다: {str(e)}"
            }
        )

@router.get("/{task_id}/pdf")
async def download_pdf(task_id: str):
    """PDF 보고서 다운로드"""
    pdf_path = get_pdf_path(task_id)
    
    if not pdf_path:
        # PDF가 아직 생성되지 않았거나 존재하지 않는 경우
        status = get_evaluation_status(task_id)
        if not status:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "TASK_NOT_FOUND",
                    "message": "해당 작업을 찾을 수 없습니다."
                }
            )
        elif status.get("status") == "failed":
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "EVALUATION_FAILED",
                    "message": status.get("error_message", "평가 중 오류가 발생했습니다.")
                }
            )
        else:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "PDF_NOT_READY",
                    "message": "PDF가 아직 생성되지 않았습니다."
                }
            )
    
    if not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "PDF_FILE_NOT_FOUND",
                "message": "PDF 파일을 찾을 수 없습니다."
            }
        )
    
    # 파일명 추출
    filename = os.path.basename(pdf_path)
    
    return FileResponse(
        path=pdf_path,
        filename=filename,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )
