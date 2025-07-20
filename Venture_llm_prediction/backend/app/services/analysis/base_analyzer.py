# app/services/analysis/base_analyzer.py
from abc import ABC, abstractmethod
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ...models.schemas import AnalysisResult, AgentState
from ...utils.config import Config

class BaseAnalyzer(ABC):
    """분석기 기본 클래스"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL, 
            temperature=Config.LLM_TEMPERATURE
        )
        self.parser = JsonOutputParser(pydantic_object=AnalysisResult)
    
    @abstractmethod
    def get_checklist(self) -> list:
        """체크리스트 반환"""
        pass
    
    @abstractmethod
    def get_prompt_template(self) -> str:
        """프롬프트 템플릿 반환"""
        pass
    
    def create_context(self, startup_name: str) -> dict:
        """컨텍스트 생성 (API 호출 등)"""
        return {}
    
    def analyze(self, state: AgentState) -> AgentState:
        """분석 실행"""
        startup_name = state.get("startup_name", "")
        if not startup_name:
            return self._handle_empty_name(state)
        
        # 컨텍스트 생성
        context = self.create_context(startup_name)
        
        # 프롬프트 생성
        prompt = ChatPromptTemplate.from_template(
            self.get_prompt_template() + "\n\n" + 
            "응답은 다음 JSON 형식으로 해주세요:\n{format_instructions}"
        )
        
        # LLM 호출
        chain = prompt | self.llm | self.parser
        
        try:
            result = chain.invoke({
                "startup_name": startup_name,
                "checklist": self.get_checklist(),
                "format_instructions": self.parser.get_format_instructions(),
                **context
            })
            
            # result가 AnalysisResult 객체인지 확인
            if not isinstance(result, AnalysisResult):
                print(f"경고: result가 AnalysisResult가 아닙니다. 타입: {type(result)}")
                # dict를 AnalysisResult로 변환 시도
                if isinstance(result, dict):
                    try:
                        result = AnalysisResult(**result)
                    except Exception as e:
                        print(f"AnalysisResult 변환 실패: {e}")
                        raise e
            
            return self._update_state(state, result)
            
        except Exception as e:
            print(f"JSON 파싱 실패, 텍스트 파싱으로 대체: {e}")
            # JSON 파싱 실패 시 텍스트 응답으로 대체
            try:
                # LLM을 다시 호출하되 JSON 파서 없이
                simple_prompt = ChatPromptTemplate.from_template(
                    self.get_prompt_template() + "\n\n" +
                    "응답 끝에 '총점: 숫자' 형태로 점수를 포함해주세요."
                )
                
                simple_chain = simple_prompt | self.llm
                text_response = simple_chain.invoke({
                    "startup_name": startup_name,
                    "checklist": self.get_checklist(),
                    **context
                })
                
                # 텍스트에서 점수 추출
                score = self._extract_score_from_text(text_response.content)
                summary = text_response.content
                
                # 임시 AnalysisResult 객체 생성
                temp_result = AnalysisResult(
                    items=[{"question": "임시", "score": score, "reasoning": summary, "sources": []}],
                    total_score=score,
                    summary=summary
                )
                
                return self._update_state(state, temp_result)
                
            except Exception as fallback_error:
                print(f"텍스트 파싱도 실패: {fallback_error}")
                return self._handle_error(state)
    
    def _extract_score_from_text(self, text: str) -> int:
        """텍스트에서 점수 추출"""
        patterns = [
            r"총점[:：]?\s*(\d{1,3})",
            r"Score[:：]?\s*(\d{1,3})",
            r"점수[:：]?\s*(\d{1,3})",
            r"(\d{1,3})\s*점",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                return min(100, max(0, score))  # 0-100 범위로 제한
        
        # 점수를 찾을 수 없으면 기본값
        return 50
    
    @abstractmethod
    def _update_state(self, state: AgentState, result: AnalysisResult) -> AgentState:
        """상태 업데이트"""
        pass
    
    @abstractmethod
    def _handle_empty_name(self, state: AgentState) -> AgentState:
        """빈 이름 처리"""
        pass
    
    @abstractmethod
    def _handle_error(self, state: AgentState) -> AgentState:
        """에러 처리"""
        pass