# app/services/analysis/growth_analyzer.py
from .base_analyzer import BaseAnalyzer
from ..external.api_client import ExternalAPIClient
from ...utils.constants import GROWTH_CHECKLIST
from ...models.schemas import AgentState, AnalysisResult

class GrowthAnalyzer(BaseAnalyzer):
    """성장률 분석기"""
    
    def __init__(self):
        super().__init__()
        self.api_client = ExternalAPIClient()
    
    def get_checklist(self) -> list:
        return GROWTH_CHECKLIST
    
    def get_prompt_template(self) -> str:
        return """
당신은 스타트업 '{startup_name}'의 성장률을 평가하는 전문가입니다.

다음 정보를 참고하여 분석하세요:
Tavily 검색 결과: {tavily_context}
Naver 뉴스 검색 결과: {naver_context}

체크리스트: {checklist}

각 항목을 0-10점으로 평가하고, 성장 지표와 트렌드를 중점적으로 분석하세요.
        """
    
    def create_context(self, startup_name: str) -> dict:
        """컨텍스트 생성"""
        tavily_results = self.api_client.search_tavily(startup_name, limit=5)
        naver_results = self.api_client.search_naver_news(startup_name, display=5)
        
        tavily_context = "\n".join([
            f"{i+1}. {item.get('title', '제목없음')} ({item.get('url')})"
            for i, item in enumerate(tavily_results)
        ]) or "정보 없음"
        
        naver_context = "\n".join([
            f"{i+1}. {item.get('title', '제목없음')} – {item.get('description', '')} ({item.get('originallink')})"
            for i, item in enumerate(naver_results)
        ]) or "정보 없음"
        
        items_formatted = "\n".join(f"{i+1}. {q}" for i, q in enumerate(self.get_checklist()))
        
        return {
            "tavily_context": tavily_context,
            "naver_context": naver_context,
            "items": items_formatted
        }
    
    def _update_state(self, state: AgentState, result: AnalysisResult) -> AgentState:
        total_score = result.get_total_score()
        state["성장률_점수"] = total_score
        state["성장률_분석_근거"] = result.summary
        print(f"성장률 분석 완료 - 점수: {total_score}")
        return state
    
    def _handle_empty_name(self, state: AgentState) -> AgentState:
        state["성장률_점수"] = 0
        state["성장률_분석_근거"] = "스타트업 이름이 제공되지 않았습니다."
        return state
    
    def _handle_error(self, state: AgentState) -> AgentState:
        state["성장률_점수"] = 0
        state["성장률_분석_근거"] = "분석 중 오류가 발생했습니다."
        return state