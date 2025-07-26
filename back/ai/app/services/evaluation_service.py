import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import os
import sys

# 같은 디렉토리의 모듈 import
from .startup_invest_evaluation import (
    analyze_product, analyze_technology, analyze_growth, 
    internal_judgement, analyze_market, analyze_competitor,
    final_judgement, generate_report, generate_pdf,
    AgentState
)

# 전역 상태 저장소 (실제 운영에서는 Redis 사용 권장)
task_status_store: Dict[str, Dict[str, Any]] = {}

# 평가 단계 정의
EVALUATION_STEPS = [
    "상품 분석",
    "기술 분석", 
    "성장성 분석",
    "내부 판단",
    "시장 분석",
    "경쟁사 분석",
    "최종 판단",
    "보고서 생성",
    "PDF 생성"
]

def create_task_id() -> str:
    """고유한 task_id 생성"""
    return f"task_{uuid.uuid4().hex[:12]}"

def update_task_status(task_id: str, **kwargs):
    """작업 상태 업데이트"""
    if task_id not in task_status_store:
        task_status_store[task_id] = {}
    
    task_status_store[task_id].update(kwargs)
    task_status_store[task_id]["updated_at"] = datetime.now().isoformat()
    
    # 디버깅을 위한 로그 출력 (주석처리)
    # if "progress" in kwargs:
    #     print(f"상태 업데이트: {task_id} - 진행률: {kwargs['progress']}%, 단계: {kwargs.get('current_step', 'N/A')}")

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """작업 상태 조회"""
    return task_status_store.get(task_id)

def calculate_progress(current_step: str) -> int:
    """현재 단계에 따른 진행률 계산"""
    try:
        step_index = EVALUATION_STEPS.index(current_step)
        # 각 단계별 진행률을 더 세밀하게 계산
        base_progress = int((step_index / len(EVALUATION_STEPS)) * 100)
        # 최소 10%씩은 진행되도록 보장
        progress = max(base_progress, 10)
        # print(f"진행률 계산: {current_step} -> {progress}%")
        return progress
    except ValueError:
        print(f"알 수 없는 단계: {current_step}, 사용 가능한 단계: {EVALUATION_STEPS}")
        return 0

def get_step_description(step: str) -> str:
    """단계별 설명 반환"""
    descriptions = {
        "상품 분석": "상품/서비스의 경쟁력을 분석하고 있습니다...",
        "기술 분석": "기술 수준과 차별성을 분석하고 있습니다...",
        "성장성 분석": "성장 가능성과 시장 트렌드 적합성을 분석하고 있습니다...",
        "내부 판단": "내부 항목들을 종합하여 판단하고 있습니다...",
        "시장 분석": "시장성과 산업 전망을 분석하고 있습니다...",
        "경쟁사 분석": "경쟁사와의 비교 분석을 진행하고 있습니다...",
        "최종 판단": "최종 투자 여부를 판단하고 있습니다...",
        "보고서 생성": "분석 결과를 종합하여 보고서를 생성하고 있습니다...",
        "PDF 생성": "PDF 보고서를 생성하고 있습니다..."
    }
    return descriptions.get(step, "분석을 진행하고 있습니다...")

async def run_evaluation_pipeline(startup_name: str, task_id: str):
    """평가 파이프라인 실행 (비동기)"""
    try:
        # 초기 상태 설정
        initial_state = AgentState(startup_name=startup_name)
        update_task_status(task_id, 
                          status="in_progress",
                          current_step="상품 분석",
                          progress=calculate_progress("상품 분석"),
                          description=get_step_description("상품 분석"))

        # 1. 상품 분석
        state = analyze_product(initial_state)
        update_task_status(task_id, 
                          current_step="기술 분석",
                          progress=15,
                          description=get_step_description("기술 분석"))

        # 2. 기술 분석
        state = analyze_technology(state)
        update_task_status(task_id, 
                          current_step="성장성 분석",
                          progress=30,
                          description=get_step_description("성장성 분석"))

        # 3. 성장성 분석
        state = analyze_growth(state)
        update_task_status(task_id, 
                          current_step="내부 판단",
                          progress=45,
                          description=get_step_description("내부 판단"))

        # 4. 내부 판단
        state = internal_judgement(state)
        
        # 내부 판단 결과에 따른 분기
        if state.get("최종_판단") == "보류":
            # 외부 항목 생략하고 바로 보고서 생성
            update_task_status(task_id, 
                              current_step="보고서 생성",
                              progress=80,
                              description=get_step_description("보고서 생성"))
            
            state = generate_report(state)
            
            update_task_status(task_id, 
                              current_step="PDF 생성",
                              progress=90,
                              description=get_step_description("PDF 생성"))
            
            try:
                state = generate_pdf(state)
            except Exception as pdf_error:
                print(f"PDF 생성 실패: {pdf_error}")
                state["pdf_path"] = ""
        else:
            # 외부 항목 분석 진행
            update_task_status(task_id, 
                              current_step="시장 분석",
                              progress=55,
                              description=get_step_description("시장 분석"))
            
            state = analyze_market(state)
            
            update_task_status(task_id, 
                              current_step="경쟁사 분석",
                              progress=65,
                              description=get_step_description("경쟁사 분석"))
            
            state = analyze_competitor(state)
            
            update_task_status(task_id, 
                              current_step="최종 판단",
                              progress=75,
                              description=get_step_description("최종 판단"))
            
            state = final_judgement(state)
            
            update_task_status(task_id, 
                              current_step="보고서 생성",
                              progress=85,
                              description=get_step_description("보고서 생성"))
            
            state = generate_report(state)
            
            update_task_status(task_id, 
                              current_step="PDF 생성",
                              progress=95,
                              description=get_step_description("PDF 생성"))
            
            try:
                state = generate_pdf(state)
            except Exception as pdf_error:
                print(f"PDF 생성 실패: {pdf_error}")
                state["pdf_path"] = ""

        # 최종 결과 저장
        scores = {
            "product": state.get("상품_점수", 0),
            "technology": state.get("기술_점수", 0),
            "growth": state.get("성장률_점수", 0),
            "market": state.get("시장성_점수", 0),
            "competition": state.get("경쟁사_점수", 0)
        }
        
        analysis_details = {
            "product_analysis": state.get("상품_분석_근거", ""),
            "technology_analysis": state.get("기술_분석_근거", ""),
            "growth_analysis": state.get("성장률_분석_근거", ""),
            "market_analysis": state.get("시장성_분석_근거", ""),
            "competition_analysis": state.get("경쟁사_분석_근거", "")
        }
        
        # 0이 아닌 점수들만 평균 계산
        valid_scores = [score for score in scores.values() if score > 0]
        if valid_scores:
            overall_score = sum(valid_scores) // len(valid_scores)
        else:
            overall_score = 0
        
        final_result = {
            "startup_name": startup_name,
            "decision": state.get("최종_판단", "보류"),
            "overall_score": overall_score,
            "scores": scores,
            "analysis_details": analysis_details,
            "pdf_path": state.get("pdf_path", ""),
            "created_at": datetime.now().isoformat()
        }
        
        # print(f"최종 결과 저장: {task_id}")
        # print(f"결과 내용: {final_result}")
        
        update_task_status(task_id,
                          status="completed",
                          current_step="완료",
                          progress=100,
                          description="분석이 완료되었습니다.",
                          result=final_result)

    except Exception as e:
        # 오류 발생 시 상태 업데이트
        update_task_status(task_id,
                          status="failed",
                          error_message=str(e),
                          description="분석 중 오류가 발생했습니다.")

async def start_evaluation(startup_name: str) -> str:
    """평가 시작"""
    task_id = create_task_id()
    
    # 초기 상태 설정
    update_task_status(task_id,
                      startup_name=startup_name,
                      status="pending",
                      current_step="대기 중",
                      progress=0,
                      description="평가를 시작합니다...",
                      created_at=datetime.now().isoformat())
    
    # 비동기로 평가 파이프라인 실행
    asyncio.create_task(run_evaluation_pipeline(startup_name, task_id))
    
    return task_id

def get_evaluation_status(task_id: str) -> Optional[Dict[str, Any]]:
    """평가 상태 조회"""
    return get_task_status(task_id)

def get_evaluation_result(task_id: str) -> Optional[Dict[str, Any]]:
    """평가 결과 조회"""
    # print(f"결과 조회 요청: {task_id}")
    status = get_task_status(task_id)
    # print(f"상태 정보: {status}")
    
    if status and status.get("status") == "completed":
        result = status.get("result")
        if result:
            # print(f"결과 반환: {result}")
            return result
        else:
            # print(f"결과가 없습니다. task_id: {task_id}, status: {status}")
            return None
    else:
        # print(f"완료되지 않은 작업입니다. task_id: {task_id}, status: {status}")
        return None

def get_pdf_path(task_id: str) -> Optional[str]:
    """PDF 파일 경로 조회"""
    result = get_evaluation_result(task_id)
    if result:
        return result.get("pdf_path")
    return None 