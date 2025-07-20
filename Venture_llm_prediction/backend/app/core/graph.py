# backend/app/core/graph.py
from langgraph.graph import StateGraph, END
from ..models.schemas import AgentState
from ..services.analysis.product_analyzer import ProductAnalyzer
from ..services.analysis.technology_analyzer import TechnologyAnalyzer
from ..services.analysis.growth_analyzer import GrowthAnalyzer
from ..services.analysis.market_analyzer import MarketAnalyzer
from ..services.analysis.competitor_analyzer import CompetitorAnalyzer
from ..services.report.report_generator import ReportGenerator
from ..services.report.pdf_generator import PDFGenerator
from .judgement import JudgementService

class AnalysisGraph:
    """분석 워크플로우 그래프"""
    
    def __init__(self):
        self.product_analyzer = ProductAnalyzer()
        self.technology_analyzer = TechnologyAnalyzer()
        self.growth_analyzer = GrowthAnalyzer()
        self.market_analyzer = MarketAnalyzer()
        self.competitor_analyzer = CompetitorAnalyzer()
        self.report_generator = ReportGenerator()
        self.pdf_generator = PDFGenerator()
        self.judgement_service = JudgementService()
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """그래프 구성"""
        graph = StateGraph(AgentState)

        # 노드 추가
        graph.add_node("AnalyzeProduct", self.product_analyzer.analyze)
        graph.add_node("AnalyzeTechnology", self.technology_analyzer.analyze)
        graph.add_node("AnalyzeGrowth", self.growth_analyzer.analyze)
        graph.add_node("InternalJudgement", self.judgement_service.internal_judgement)
        graph.add_node("AnalyzeMarket", self.market_analyzer.analyze)
        graph.add_node("AnalyzeCompetitor", self.competitor_analyzer.analyze)
        graph.add_node("FinalJudgement", self.judgement_service.final_judgement)
        graph.add_node("GenerateReport", self.report_generator.generate_report)
        graph.add_node("GeneratePDF", self.pdf_generator.generate_pdf)

        # 엣지 설정
        graph.set_entry_point("AnalyzeProduct")
        graph.add_edge("AnalyzeProduct", "AnalyzeTechnology")
        graph.add_edge("AnalyzeTechnology", "AnalyzeGrowth")
        graph.add_edge("AnalyzeGrowth", "InternalJudgement")

        # 조건부 엣지
        graph.add_conditional_edges(
            "InternalJudgement",
            self._route_after_internal_judgement
        )

        graph.add_edge("AnalyzeMarket", "AnalyzeCompetitor")
        graph.add_edge("AnalyzeCompetitor", "FinalJudgement")
        graph.add_edge("FinalJudgement", "GenerateReport")
        graph.add_edge("GenerateReport", "GeneratePDF")
        graph.add_edge("GeneratePDF", END)

        return graph.compile()
    
    def _route_after_internal_judgement(self, state: AgentState) -> str:
        """내부 판단 후 라우팅"""
        if state.get("최종_판단") == "보류":
            return "GenerateReport"
        return "AnalyzeMarket"
    
    def analyze_startup(self, startup_name: str) -> AgentState:
        """스타트업 분석 실행"""
        initial_state = {"startup_name": startup_name}
        return self.graph.invoke(initial_state)