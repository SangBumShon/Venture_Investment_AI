# app/services/report/report_generator.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ...models.schemas import AgentState
from ...utils.config import Config

class ReportGenerator:
    """보고서 생성 서비스"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE
        )
    
    def generate_report(self, state: AgentState) -> AgentState:
        """투자 심사 보고서 생성"""
        startup_name = state.get("startup_name", "알 수 없는 스타트업")
        
        prompt = ChatPromptTemplate.from_template(
            "스타트업 '{startup_name}'에 대한 투자 심사 결과는 {최종_판단} 입니다. 점수 요약:\n"
            "- 상품/서비스: {상품_점수}\n"
            "- 기술: {기술_점수}\n"
            "- 성장률: {성장률_점수}\n"
            "- 시장성: {시장성_점수}\n"
            "- 경쟁사: {경쟁사_점수}\n\n"
            "각 항목별 분석 근거:\n"
            "1. 상품/서비스 분석:\n{상품_분석_근거}\n\n"
            "2. 기술 분석:\n{기술_분석_근거}\n\n"
            "3. 성장률 분석:\n{성장률_분석_근거}\n\n"
            "4. 시장성 분석:\n{시장성_분석_근거}\n\n"
            "5. 경쟁사 분석:\n{경쟁사_분석_근거}\n\n"
            "위 분석 결과를 바탕으로 투자 심사 보고서를 작성하세요. 각 항목별 강점과 약점을 요약하고, "
            "최종 판단의 근거를 명확히 제시하며, 개선이 필요한 부분에 대한 제안도 포함해주세요."
        )
        
        chain = prompt | self.llm
        report = chain.invoke({
            "startup_name": startup_name,
            "최종_판단": state.get("최종_판단", "보류"),
            "상품_점수": state.get("상품_점수", 0),
            "기술_점수": state.get("기술_점수", 0),
            "성장률_점수": state.get("성장률_점수", 0),
            "시장성_점수": state.get("시장성_점수", 0),
            "경쟁사_점수": state.get("경쟁사_점수", 0),
            "상품_분석_근거": state.get("상품_분석_근거", "정보 없음"),
            "기술_분석_근거": state.get("기술_분석_근거", "정보 없음"),
            "성장률_분석_근거": state.get("성장률_분석_근거", "정보 없음"),
            "시장성_분석_근거": state.get("시장성_분석_근거", "정보 없음"),
            "경쟁사_분석_근거": state.get("경쟁사_분석_근거", "정보 없음")
        })
        
        state["보고서"] = report.content
        print("보고서 생성 완료")
        return state
