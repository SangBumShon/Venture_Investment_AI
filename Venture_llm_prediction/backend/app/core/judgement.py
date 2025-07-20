# backend/app/core/judgement.py
from ..models.schemas import AgentState

class JudgementService:
    """판단 로직 서비스"""
    
    @staticmethod
    def internal_judgement(state: AgentState) -> AgentState:
        """내부 판단 (1차 심사)"""
        if (
            state["상품_점수"] < 40 or
            state["기술_점수"] < 40 or
            state["성장률_점수"] < 40
        ):
            state["최종_판단"] = "보류"
        elif (
            (state["상품_점수"] + state["기술_점수"] + state["성장률_점수"]) / 3 < 60
        ):
            state["최종_판단"] = "보류"
        
        print(f"내부 판단 완료 - 결과: {state.get('최종_판단', '진행')}")
        return state
    
    @staticmethod
    def final_judgement(state: AgentState) -> AgentState:
        """최종 판단"""
        avg_internal = (state["상품_점수"] + state["기술_점수"] + state["성장률_점수"]) / 3
        avg_total = (avg_internal + state["시장성_점수"] + state["경쟁사_점수"]) / 3
        state["최종_판단"] = "투자" if avg_total >= 65 else "보류"
        print(f"최종 판단 완료 - 결과: {state['최종_판단']}")
        return state

