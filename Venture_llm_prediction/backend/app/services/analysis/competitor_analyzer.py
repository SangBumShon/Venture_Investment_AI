# app/services/analysis/competitor_analyzer.py
from .base_analyzer import BaseAnalyzer
from ..external.api_client import ExternalAPIClient
from ...utils.constants import COMPETITOR_CHECKLIST
from ...models.schemas import AgentState, AnalysisResult

class CompetitorAnalyzer(BaseAnalyzer):
    """경쟁사 분석기"""
    
    def __init__(self):
        super().__init__()
        self.api_client = ExternalAPIClient()
    
    def get_checklist(self) -> list:
        return COMPETITOR_CHECKLIST
    
    def get_prompt_template(self) -> str:
        return """
당신은 스타트업 '{startup_name}'의 경쟁사를 평가하는 전문가입니다.

다음 정보를 참고하여 분석하세요:
웹 검색 결과: {web_context}

체크리스트: {checklist}

각 항목을 0-10점으로 평가하고, 경쟁 환경과 차별화 요소를 중점적으로 분석하세요.
        """
    
    def create_context(self, startup_name: str) -> dict:
        """컨텍스트 생성"""
        web_results = self.api_client.search_tavily(
            f"{startup_name} 경쟁사 AI 스타트업 시장 분석", 
            limit=20
        )
        
        web_context = "\n".join([
            f"{i+1}. {item.get('title', '제목없음')} ({item.get('url')})"
            for i, item in enumerate(web_results)
        ]) or "정보 없음"
        
        items_formatted = "\n".join(f"{i+1}. {q}" for i, q in enumerate(self.get_checklist()))
        
        return {
            "web_context": web_context,
            "items": items_formatted
        }
    
    def _update_state(self, state: AgentState, result: AnalysisResult) -> AgentState:
        total_score = result.get_total_score()
        state["경쟁사_점수"] = total_score
        state["경쟁사_분석_근거"] = result.summary
        print(f"경쟁사 분석 완료 - 점수: {total_score}")
        return state
    
    def _handle_empty_name(self, state: AgentState) -> AgentState:
        state["경쟁사_점수"] = 0
        state["경쟁사_분석_근거"] = "스타트업 이름이 제공되지 않았습니다."
        return state
    
    def _handle_error(self, state: AgentState) -> AgentState:
        state["경쟁사_점수"] = 0
        state["경쟁사_분석_근거"] = "분석 중 오류가 발생했습니다."
        return state
