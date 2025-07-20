# app/core/orchestrator.py
from .graph import AnalysisGraph
from ..models.schemas import StartupAnalysisRequest, StartupAnalysisResponse, AgentState
from ..utils.config import Config
import datetime
from typing import Dict, Any

class AnalysisOrchestrator:
    """분석 오케스트레이션 서비스"""
    
    def __init__(self):
        Config.create_directories()
        self.analysis_graph = AnalysisGraph()
    
    async def analyze_startup(self, request: StartupAnalysisRequest) -> StartupAnalysisResponse:
        """스타트업 분석 실행"""
        try:
            # 분석 실행
            result = self.analysis_graph.analyze_startup(request.startup_name)
            
            # 응답 생성
            response = StartupAnalysisResponse(
                startup_name=request.startup_name,
                final_decision=result.get("최종_판단", "보류"),
                scores={
                    "상품/서비스": result.get("상품_점수", 0),
                    "기술": result.get("기술_점수", 0),
                    "성장률": result.get("성장률_점수", 0),
                    "시장성": result.get("시장성_점수", 0),
                    "경쟁사": result.get("경쟁사_점수", 0)
                },
                report_path=result.get("pdf_path", ""),
                created_at=datetime.datetime.now().isoformat()
            )
            
            return response
            
        except Exception as e:
            print(f"분석 중 오류 발생: {e}")
            raise e
    
    def get_analysis_status(self, startup_name: str) -> Dict[str, Any]:
        """분석 상태 조회 (향후 비동기 처리용)"""
        # 향후 Redis 등을 사용한 상태 관리
        return {"status": "completed", "startup_name": startup_name}
