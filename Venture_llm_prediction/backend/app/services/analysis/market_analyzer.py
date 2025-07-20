# app/services/analysis/market_analyzer.py
from .base_analyzer import BaseAnalyzer
from ..external.rag_service import RAGService
from ..external.api_client import ExternalAPIClient
from ...utils.constants import MARKET_CHECKLIST
from ...models.schemas import AgentState, AnalysisResult

class MarketAnalyzer(BaseAnalyzer):
    """시장성 분석기"""
    
    def __init__(self):
        super().__init__()
        self.rag_service = RAGService()
        self.api_client = ExternalAPIClient()
    
    def get_checklist(self) -> list:
        return MARKET_CHECKLIST
    
    def get_prompt_template(self) -> str:
        return """
당신은 스타트업 '{startup_name}'의 시장성을 평가하는 전문가입니다.

다음 정보를 종합하여 분석하세요:
PDF 기반 RAG 검색 결과: {rag_context}
웹 검색 결과: {web_context}

체크리스트: {checklist}

각 항목을 0-10점으로 평가하고, 시장 규모와 성장성을 중점적으로 분석하세요.
        """
    
    def create_context(self, startup_name: str) -> dict:
        """컨텍스트 생성"""
        # RAG 검색
        rag_context = self.rag_service.search_market_info(startup_name)
        
        # 웹 검색
        web_results = self.api_client.search_tavily(
            f"{startup_name} AI 스타트업 시장성 시장 규모 성장성", 
            limit=10
        )
        web_context = "\n".join([
            f"{i+1}. {item.get('title', '제목없음')} ({item.get('url')})"
            for i, item in enumerate(web_results)
        ]) or "정보 없음"
        
        items_formatted = "\n".join(f"{i+1}. {q}" for i, q in enumerate(self.get_checklist()))
        
        return {
            "rag_context": rag_context,
            "web_context": web_context,
            "items": items_formatted
        }
    
    def _update_state(self, state: AgentState, result: AnalysisResult) -> AgentState:
        total_score = result.get_total_score()
        state["시장성_점수"] = total_score
        state["시장성_분석_근거"] = result.summary
        print(f"시장성 분석 완료 - 점수: {total_score}")
        return state
    
    def _handle_empty_name(self, state: AgentState) -> AgentState:
        state["시장성_점수"] = 0
        state["시장성_분석_근거"] = "스타트업 이름이 제공되지 않았습니다."
        return state
    
    def _handle_error(self, state: AgentState) -> AgentState:
        state["시장성_점수"] = 0
        state["시장성_분석_근거"] = "분석 중 오류가 발생했습니다."
        return state