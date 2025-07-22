# %%
# !pip install -r requirements.txt

# %%
from dotenv import load_dotenv

load_dotenv()

# %%
# from langchain_teddynote import logging
# logging.langsmith("AI-project")

# %%
from typing import TypedDict, Literal, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from utils import (
    search_tavily, search_naver_news, get_analysis_score, 
    extract_insufficient_items, format_checklist_items, create_analysis_prompt_template
)

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
import markdown2
import pdfkit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import datetime
import os
import re
import requests

# %%
# 1. 상태 스키마 정의 (모든 Agent가 공유)
class AgentState(TypedDict, total=False):
    startup_name: str  # 스타트업 이름 추가
    상품_점수: int
    기술_점수: int
    성장률_점수: int
    시장성_점수: int
    경쟁사_점수: int
    최종_판단: Literal["투자", "보류"]
    보고서: str
    pdf_path: str  # PDF 파일 경로 추가
    # 분석 근거 추가
    상품_분석_근거: str
    기술_분석_근거: str
    성장률_분석_근거: str
    시장성_분석_근거: str
    경쟁사_분석_근거: str

# %%
# 2. LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# %%

def analyze_product(state: AgentState) -> AgentState:
    startup_name = state.get("startup_name", "")
    if not startup_name:
        state["상품_점수"] = 0
        state["상품_분석_근거"] = "스타트업 이름이 제공되지 않았습니다."
        return state

    try:
        # API 키 검증
        from utils import validate_api_keys
        validate_api_keys(["tavily_key", "naver_id", "naver_secret"])
    except ValueError as e:
        state["상품_점수"] = 0
        state["상품_분석_근거"] = str(e)
        return state

    checklist = [
        "제품이 명확한 문제를 해결하는가?",
        "제품 기능이 사용자가 기대하는 가치를 제공하는가?",
        "제품의 차별화 요소가 명확한가?",
        "제품의 사용성(UI/UX)이 직관적인가?",
        "제품의 기술적 구현 가능성이 높은가?",
        "제품의 시장 수요가 충분한가?",
        "제품 가격 전략이 합리적인가?",
        "제품 출시 및 확장 계획이 구체적인가?",
        "경쟁 제품 대비 우위가 있는가?",
        "고객 피드백 수집 및 반영 체계가 갖춰져 있는가?"
    ]
    items_formatted = format_checklist_items(checklist)

    # Tavily & Naver 검색
    tavily_context = search_tavily(startup_name, 5)
    naver_context = search_naver_news(startup_name, 5)

    # LLM Prompt
    prompt_template = create_analysis_prompt_template(checklist)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    response = (prompt | llm).invoke({
        "startup_name": startup_name,
        "items": items_formatted,
        "tavily_context": tavily_context,
        "naver_context": naver_context
    })

    # 결과 파싱
    analysis = response.content
    score = get_analysis_score(analysis, checklist)

    state["상품_점수"] = score
    state["상품_분석_근거"] = analysis
    print(state["상품_점수"])
    print(state["상품_분석_근거"])
    return state



# 메인 함수
def analyze_technology(state: "AgentState") -> "AgentState":
    startup_name = state.get("startup_name", "")
    if not startup_name:
        state["기술_점수"] = 0
        state["기술_분석_근거"] = "스타트업 이름이 제공되지 않았습니다."
        return state

    try:
        # API 키 검증
        from utils import validate_api_keys
        validate_api_keys(["tavily_key", "naver_id", "naver_secret"])
    except ValueError as e:
        state["기술_점수"] = 0
        state["기술_분석_근거"] = str(e)
        return state

    checklist = [
        "기술적 차별성", "특허 보유 여부", "스케일링 가능성", "기술 성숙도",
        "인력 역량", "기술 난이도", "기술 구현 가능성", "기술 유지보수 용이성",
        "기술 표준 준수 여부", "기술 관련 외부 인증 또는 수상 이력"
    ]
    items_formatted = format_checklist_items(checklist)

    prompt_template = create_analysis_prompt_template(checklist)
    prompt = ChatPromptTemplate.from_template(prompt_template)

    analysis = ""
    for attempt in range(1, 4):
        # Tavily & Naver 검색
        tavily_context = search_tavily(startup_name, 5)

        # Naver 검색
        naver_context = search_naver_news(startup_name, 5)

        # LLM 호출
        response = (prompt | llm).invoke({
            "startup_name": startup_name,
            "items": items_formatted,
            "tavily_context": tavily_context,
            "naver_context": naver_context
        })
        analysis = response.content

        insufficient = extract_insufficient_items(analysis, checklist)
        if len(insufficient) < 5 or attempt == 3:
            break

    # robust 점수 파싱
    score = get_analysis_score(analysis, checklist)

    state["기술_점수"] = score
    state["기술_분석_근거"] = analysis
    return state



def analyze_growth(state: "AgentState") -> "AgentState":
    startup_name = state.get("startup_name", "")
    if not startup_name:
        state["성장률_점수"] = 0
        state["성장률_분석_근거"] = "스타트업 이름이 제공되지 않았습니다."
        return state

    try:
        # API 키 검증
        from utils import validate_api_keys
        validate_api_keys(["tavily_key", "naver_id", "naver_secret"])
    except ValueError as e:
        state["성장률_점수"] = 0
        state["성장률_분석_근거"] = str(e)
        return state

    checklist = [
        "매출 성장률", "사용자 증가율", "시장 점유율 변화", "고객 유지율 (Retention Rate)",
        "월간/분기별 활성 사용자 증가 (MAU/WAU)", "신규 계약/클라이언트 수 증가", "연간 반복 매출(ARR) 성장",
        "투자 유치 규모 변화", "직원 수 증가율", "해외/신시장 진출 속도"
    ]
    items_formatted = format_checklist_items(checklist)

    prompt_template = create_analysis_prompt_template(checklist)
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Tavily & Naver 검색
    tavily_context = search_tavily(startup_name, 5)
    naver_context = search_naver_news(startup_name, 5)

    response = (prompt | llm).invoke({
        "startup_name": startup_name,
        "items": items_formatted,
        "tavily_context": tavily_context,
        "naver_context": naver_context
    })
    analysis = response.content

    # 점수 robust 파싱
    score = get_analysis_score(analysis, checklist)

    state["성장률_점수"] = score
    state["성장률_분석_근거"] = analysis
    return state


def internal_judgement(state: AgentState) -> AgentState:
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
    return state

def compare_retrieval_methods(all_docs, query):
    """BM25, Cosine Similarity, Hybrid 방식의 성능을 비교하는 함수"""
    import time
    
    results = {}
    
    # 1. BM25 검색
    print("\n[성능 비교] BM25 검색 시작...")
    start_time = time.time()
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 5
    bm25_docs = bm25_retriever.get_relevant_documents(query)
    bm25_time = time.time() - start_time
    
    results['bm25'] = {
        'docs': bm25_docs,
        'time': bm25_time,
        'count': len(bm25_docs)
    }
    print(f"BM25 검색 완료: {bm25_time:.3f}초, {len(bm25_docs)}개 문서")
    
    # 2. Cosine Similarity 검색
    print("\n[성능 비교] Cosine Similarity 검색 시작...")
    start_time = time.time()
    vector_store = Chroma.from_documents(all_docs, OpenAIEmbeddings())
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    vector_docs = vector_retriever.get_relevant_documents(query)
    vector_time = time.time() - start_time
    
    results['cosine'] = {
        'docs': vector_docs,
        'time': vector_time,
        'count': len(vector_docs)
    }
    print(f"Cosine Similarity 검색 완료: {vector_time:.3f}초, {len(vector_docs)}개 문서")
    
    # 3. Hybrid 검색
    print("\n[성능 비교] Hybrid 검색 시작...")
    start_time = time.time()
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    hybrid_docs = ensemble_retriever.get_relevant_documents(query)
    hybrid_time = time.time() - start_time
    
    results['hybrid'] = {
        'docs': hybrid_docs,
        'time': hybrid_time,
        'count': len(hybrid_docs)
    }
    print(f"Hybrid 검색 완료: {hybrid_time:.3f}초, {len(hybrid_docs)}개 문서")
    
    # 4. 성능 비교 결과 출력
    print("\n" + "="*60)
    print("🔍 검색 방식별 성능 비교 결과")
    print("="*60)
    print(f"{'방식':<15} {'소요시간':<10} {'문서수':<8} {'상대속도':<10}")
    print("-"*60)
    
    fastest_time = min(bm25_time, vector_time, hybrid_time)
    
    for method, data in results.items():
        relative_speed = fastest_time / data['time']
        method_name = {
            'bm25': 'BM25',
            'cosine': 'Cosine',
            'hybrid': 'Hybrid'
        }[method]
        print(f"{method_name:<15} {data['time']:<10.3f} {data['count']:<8} {relative_speed:<10.2f}x")
    
    # 5. 문서 중복도 분석
    print("\n📊 문서 중복도 분석:")
    all_doc_contents = []
    for method, data in results.items():
        method_docs = [doc.page_content[:100] for doc in data['docs']]
        all_doc_contents.extend(method_docs)
    
    unique_docs = len(set(all_doc_contents))
    total_docs = len(all_doc_contents)
    diversity_score = unique_docs / total_docs if total_docs > 0 else 0
    
    print(f"총 검색된 문서: {total_docs}개")
    print(f"고유 문서: {unique_docs}개")
    print(f"다양성 점수: {diversity_score:.2f} (1.0에 가까울수록 다양함)")
    
    return results

def analyze_market(state: Dict[str, Any]) -> Dict[str, Any]:
    startup_name = state.get("startup_name", "")
    if not startup_name:
        state["시장성_점수"] = 0
        state["시장성_분석_근거"] = "스타트업 이름이 제공되지 않았습니다."
        return state

    # ✅ PDF 폴더 내 전체 PDF 로드
    pdf_dir = os.path.join(os.getcwd(), "data")
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    print(f"\n[DEBUG] 폴더 내 PDF 파일 수: {len(pdf_files)}")

    all_docs = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        loader = PyMuPDFLoader(pdf_path)
        split_docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100))
        print(f"[DEBUG] '{pdf_file}' → {len(split_docs)} 청크 생성")
        all_docs.extend(split_docs)

    # 성능 비교 실행 (선택적)
    query = f"{startup_name} 시장성, 시장 규모, 성장성, 수요 동향, 트렌드"
    
    # 성능 비교를 원하면 아래 주석을 해제하세요
    # comparison_results = compare_retrieval_methods(all_docs, query)
    
    # Cosine Similarity만 사용 (벡터 검색)
    vector_store = Chroma.from_documents(all_docs, OpenAIEmbeddings())
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # 벡터 검색 실행
    retrieved_docs = vector_retriever.get_relevant_documents(query)
    print(f"\n[DEBUG] Cosine Similarity RAG 검색 결과 - 총 {len(retrieved_docs)}개")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n[벡터 검색 결과 {i+1}] (Page: {doc.metadata.get('page', '알 수 없음')})\n{doc.page_content[:300]}...")

    rag_context = "\n\n".join([f"(Page: {doc.metadata.get('page', '알 수 없음')})\n{doc.page_content}" for doc in retrieved_docs]) or "PDF에서 유의미한 정보 없음"

    # ✅ Web Search part (Tavily)
    search_tool = TavilySearchResults(k=10)
    web_results = search_tool.invoke(f"{startup_name} AI 스타트업 시장성, 시장 규모, 성장성, 수요 동향, 트렌드 최근 6개월 기사")
    web_context = "\n".join([f"{i+1}. {result['title']} ({result['url']})" for i, result in enumerate(web_results)]) or "웹 검색에서 유의미한 정보 없음"

    combined_context = f"[PDF 기반 RAG 검색 결과]\n{rag_context}\n\n[웹 검색 결과]\n{web_context}"

    print(f"\n[DEBUG] 최종 combined_context (앞 1000자):\n{combined_context[:1000]}...")

    checklist = [
        "시장 규모 및 성장성",
        "산업 내 수요 트렌드",
        "고객군 다양성 및 확보 가능성",
        "시장 진입 가능성 (규제, 장벽 등)",
        "시장 내 대체재/경쟁 제품 존재 여부",
        "향후 3~5년 성장 전망",
        "국내외 시장 확장 가능성",
        "산업 내 파트너쉽 및 생태계 가능성",
        "사회적/경제적 메가트렌드 부합 여부",
        "기술 변화에 따른 시장 위험성"
    ]

    prompt = ChatPromptTemplate.from_template(
        "당신은 AI 스타트업 시장성 평가 전문가입니다. '{startup_name}'의 시장성을 종합적으로 평가해 주세요.\n\n"
        "다음 정보를 종합하여 분석하세요:\n{combined_context}\n\n"
        "체크리스트:\n" +
        "\n".join([f"{i+1}. {q}" for i, q in enumerate(checklist)]) + "\n\n"
        "각 항목은 0점에서 10점 사이로 자유롭게 점수를 부여하세요.\n"
        "응답 형식:\n"
        "- 각 항목별 점수(0~10)와 **출처 포함한 분석 근거 (출처는 기사 제목, URL 또는 PDF 페이지 번호 명시)**\n"
        "- 결론 및 종합 분석\n"
        "- 총점: 점수 (숫자만 입력하세요, 예: 75)"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm

    response = chain.invoke({
        "startup_name": startup_name,
        "combined_context": combined_context,
    })

    analysis = response.content
    print(f"\n[DEBUG] LLM 응답 (앞 1000자):\n{analysis[:1000]}...")

    score = get_analysis_score(analysis, checklist)

    print(f"\n[DEBUG] 최종 총점: {score}")
    state["시장성_점수"] = score
    state["시장성_분석_근거"] = analysis
    return state




def analyze_competitor(state: Dict[str, Any]) -> Dict[str, Any]:
    startup_name = state.get("startup_name", "")
    if not startup_name:
        state["경쟁사_점수"] = 0
        state["경쟁사_분석_근거"] = "스타트업 이름이 제공되지 않았습니다."
        print("[DEBUG] 스타트업 이름 없음")
        return state

    search_tool = TavilySearchResults(k=20)
    search_results = search_tool.invoke(f"{startup_name} 경쟁사 AI 스타트업 시장 분석 최근 6개월 기사")

    print(f"\n[DEBUG] 검색된 기사 수: {len(search_results)}개")
    for idx, result in enumerate(search_results, 1):
        print(f"{idx}. {result['title']}")

    if len(search_results) < 10:
        print("[경고] 검색된 기사 수가 10개 미만입니다. 정보 부족으로 분석 신뢰도가 낮을 수 있습니다.")

    checklist = [
        "시장 진입 장벽 분석",
        "주요 경쟁사 식별",
        "경쟁사 제품/서비스 차별점",
        "시장 점유율 데이터",
        "가격 전략 비교",
        "기술적 우위 분석",
        "타겟 고객층 중복도",
        "성장 속도 및 추세",
        "투자 유치 상황",
        "경쟁사 대응 전략 가능성"
    ]

    prompt = ChatPromptTemplate.from_template(
        "당신은 AI 스타트업 경쟁 분석 전문가입니다. '{startup_name}'의 경쟁사 환경을 종합적으로 평가해 주세요.\n\n"
        "최근 6개월 이내에 발행된 20개 이상의 기사를 기반으로, 다음 체크리스트의 각 항목을 평가하세요.\n"
        "각 항목은 0점에서 10점 사이로 자유롭게 점수를 부여할 수 있습니다. 점수 부여 기준:\n"
        "- 매우 우수하거나 충분히 충족 → 9~10점\n"
        "- 일부 충족되었거나 불완전 → 5~8점\n"
        "- 거의 정보가 없거나 충족되지 않음 → 0~4점\n\n"
        "웹 검색 결과:\n{search_results}\n\n"
        "체크리스트:\n"
        "1. 시장 진입 장벽 분석\n"
        "2. 주요 경쟁사 식별\n"
        "3. 경쟁사 제품/서비스 차별점\n"
        "4. 시장 점유율 데이터\n"
        "5. 가격 전략 비교\n"
        "6. 기술적 우위 분석\n"
        "7. 타겟 고객층 중복도\n"
        "8. 성장 속도 및 추세\n"
        "9. 투자 유치 상황\n"
        "10. 경쟁사 대응 전략 가능성\n\n"
        "응답 형식:\n"
        "- 각 항목별 점수(0~10)와 **출처 포함한 분석 근거 (출처는 기사 제목 또는 URL 명시)**\n"
        "- 결론 및 종합 분석\n"
        "- 총점: 점수 (숫자만 입력하세요, 예: 75)"
    )

    chain = prompt | llm
    response = chain.invoke({
        "startup_name": startup_name,
        "search_results": search_results
    })

    analysis = response.content
    print("\n[DEBUG] LLM 응답 내용:")
    print(analysis)

    score = extract_total_score_from_analysis(analysis)
    if score is None:
        print("\n[DEBUG] 총점 파싱 실패 → extract_checklist_scores_competitor()에서 항목별 점수 직접 계산")
        score = extract_checklist_scores_competitor(analysis, checklist)

    state["경쟁사_점수"] = score
    state["경쟁사_분석_근거"] = analysis
    print(f"[DEBUG] 최종 총점: {score}")
    return state


def extract_total_score_from_analysis(analysis: str) -> int:
    """LLM 응답에서 다양한 총점 표현을 robust하게 파싱"""
    patterns = [
        r"\*\*총점\*\*[:：]?\s*(\d{1,3})",       # '**총점**: 69'
        r"총점[:：]?\s*(\d{1,3})\s*(?:점|/100)?",  # '총점: 69', '총점: 69점'
        r"Score[:：]?\s*(\d{1,3})\s*(?:점|/100)?",  # 'Score: 69', 'Score: 69/100'
    ]
    for pattern in patterns:
        match = re.search(pattern, analysis, re.IGNORECASE)
        if match:
            print(f"[DEBUG] 정규표현식으로 직접 파싱된 총점: {match.group(1)}")
            return int(match.group(1))
    return None

def extract_checklist_scores(analysis: str, checklist: List[str]) -> int:
    """체크리스트 항목별 점수를 유연하게 파싱 (다양한 표현 허용)"""
    # ✅ 총점 제거
    clean_analysis = re.sub(r"총점[:：]?\s*\d{1,3}\s*(?:점|/100)?", "", analysis, flags=re.IGNORECASE)

    total_score = 0
    print("\n[DEBUG] 체크리스트별 점수 파싱 시작:")
    for i, item in enumerate(checklist, 1):
        # 모든 항목 공통적으로 사용할 패턴 리스트 (가장 일반적 → 구체적 순서로)
        patterns = [
            fr"{i}\.\s.*?(\d{{1,2}})\s*/\s*10",  # '9/10'
            fr"{i}\.\s.*?(\d{{1,2}})점",         # '9점'
            fr"{i}\.\s.*?점수[:：]?\s*(\d{{1,2}})",  # '점수: 9'
            fr"{item}.*?(\d{{1,2}})\s*/\s*10",
            fr"{item}.*?(\d{{1,2}})점",
            fr"{item}.*?점수[:：]?\s*(\d{{1,2}})",
        ]
        found = False
        for pattern in patterns:
            match = re.search(pattern, clean_analysis, re.DOTALL | re.IGNORECASE)
            if match:
                item_score = int(match.group(1))
                print(f"- 항목 {i}: {item} → 점수: {item_score}")
                total_score += item_score
                found = True
                break
        if not found:
            print(f"- 항목 {i}: {item} → 점수 찾지 못함 (0점 처리)")

    print(f"[DEBUG] 체크리스트 총합 (정확한 합산): {total_score}")
    return min(100, max(0, total_score))

# %%
def final_judgement(state: AgentState) -> AgentState:
    avg_internal = (state["상품_점수"] + state["기술_점수"] + state["성장률_점수"]) / 3
    avg_total = (avg_internal + state["시장성_점수"] + state["경쟁사_점수"]) / 3
    state["최종_판단"] = "투자" if avg_total >= 65 else "보류"
    return state

# %%
def generate_report(state: AgentState) -> AgentState:
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
    chain = prompt | llm
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
    return state


# ─────────── 환경별 경로 수정 ───────────
BASE_DIR = os.getcwd()

WKHTMLTOPDF_BIN = os.path.join(BASE_DIR, "wkhtmltopdf", "bin", "wkhtmltopdf.exe")
NANUM_REG_TTF   = os.path.join(BASE_DIR, "font", "NanumGothic.ttf")
NANUM_BOLD_TTF  = os.path.join(BASE_DIR, "font", "NanumGothicBold.ttf")
# ────────────────────────────────────────

# 항상 들어갈 안내 문구
NOTICE = "상품/서비스, 기술, 성장률 중에 하나라도 40점 미만이거나 평균 60점 미만이면 시장성과 경쟁사 항목은 측정되지 않습니다."


# 1) Markdown 생성
def generate_markdown(state: dict, md_path: str) -> None:
    sn  = state.get("startup_name", "알 수 없는 스타트업")
    rep = state.get("보고서",       "보고서 내용이 없습니다.")
    dec = state.get("최종_판단",    "보류")

    labels = ["상품/서비스","기술","성장률","시장성","경쟁사"]
    keys   = ["상품_점수","기술_점수","성장률_점수","시장성_점수","경쟁사_점수"]
    scores = [int(state.get(k,0)) for k in keys]
    avg    = sum(scores)/len(scores)

    dec_html = ("<span style='color:green;'>투자</span>"
                if dec=="투자" else "<span style='color:red;'>보류</span>")

    md = [
        f"<h1 align='center'>{sn} 투자 분석 보고서</h1>", "",
        f"- **작성일** : {datetime.datetime.now():%Y년 %m월 %d일}",
        f"- **최종 판단** : {dec_html}", "", "---", "",
        "## 1. 점수 요약", "",
        "| 평가 항목 | 점수 |", "|:-----------:|:----:|",
        *[f"| {l} | **{v}** |" for l,v in zip(labels,scores)],
        f"| **평균** | **{avg:.1f}** |", "",
        f"> **{NOTICE}**",                    # 안내문 삽입
        "", "---", "",
        "## 2. 상세 분석", ""
    ]

    # 상세 분석 – 줄바꿈 로직 그대로 유지
    for line in rep.strip().splitlines():
        if   line.startswith("### "): md.append(f"#### **{line[4:]}**")
        elif line.startswith("## " ): md.append(f"### **{line[3:]}**")
        elif line.startswith("# "  ): md.append(f"## **{line[2:]}**")
        else:                         md.append(line)

    md += ["", "---", "*이 보고서는 AI 분석 시스템에 의해 자동 생성되었습니다.*"]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


# 2) Markdown → PDF (wkhtmltopdf)
def convert_md_to_pdf(md_path:str, pdf_path:str) -> None:
    cfg = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_BIN)
    with open(md_path,"r",encoding="utf-8") as f:
        body_html = markdown2.markdown(f.read(), extras=["tables"])

    if os.path.isfile(NANUM_REG_TTF):
        reg  = os.path.abspath(NANUM_REG_TTF).replace("\\","/")
        bold = os.path.abspath(NANUM_BOLD_TTF if os.path.isfile(NANUM_BOLD_TTF) else NANUM_REG_TTF).replace("\\","/")
        font_css = (f"@font-face{{font-family:'NanumGothic';src:url('file:///{reg}') format('truetype');font-weight:normal;}}"
                    f"@font-face{{font-family:'NanumGothic';src:url('file:///{bold}') format('truetype');font-weight:bold;}}")
        family = "NanumGothic"
    else:
        font_css, family = "", "sans-serif"

    style = f"""
    <style>
    {font_css}
    body{{margin:40px 50px 60px;font-family:'{family}';line-height:1.6;font-size:11pt}}
    h1{{font-size:20pt;text-align:center;margin-bottom:0.6em}}
    h2{{font-size:15pt;margin-top:1.5em;margin-bottom:0.4em}}
    table{{border-collapse:collapse;width:100%;margin-top:0.8em;font-size:10.5pt}}
    th,td{{border:1px solid #666;padding:6px 8px;text-align:center}}
    th{{background:#e0e0e0;font-weight:bold}}
    tr:last-child td{{background:#f5f5f5;font-weight:bold}}
    </style>"""

    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{style}</head><body>{body_html}</body></html>"
    pdfkit.from_string(html, pdf_path, configuration=cfg,
                       options={"enable-local-file-access":None,"encoding":"utf-8"})


# 3) ReportLab PDF + MarkdownPDF 통합
def generate_pdf(state: dict) -> dict:
    sn     = state.get("startup_name","알 수 없는 스타트업")
    today  = datetime.datetime.now().strftime("%Y%m%d")
    out    = "investment_reports"; os.makedirs(out, exist_ok=True)
    lab_pdf = os.path.join(out, f"{sn}_투자분석보고서_{today}_lab.pdf")
    md_path = os.path.join(out, f"{sn}_보고서.md")
    htmlpdf = os.path.join(out, f"{sn}_투자분석보고서_{today}.pdf")

    doc, styles = SimpleDocTemplate(lab_pdf, pagesize=letter), getSampleStyleSheet()
    try:
        pdfmetrics.registerFont(TTFont("NanumGothic",      NANUM_REG_TTF))
        pdfmetrics.registerFont(TTFont("NanumGothic-Bold", NANUM_BOLD_TTF))
        base, bold = "NanumGothic", "NanumGothic-Bold"
    except Exception:
        base, bold = "Helvetica", "Helvetica-Bold"

    styles.add(ParagraphStyle("ReportTitle",   parent=styles["Heading1"], fontSize=18, alignment=1, spaceAfter=20, fontName=base))
    styles.add(ParagraphStyle("ReportSubtitle",parent=styles["Heading2"], fontSize=14, spaceBefore=10, spaceAfter=10, fontName=base))
    styles.add(ParagraphStyle("SectionTitle",  parent=styles["Normal"],   fontSize=12, spaceBefore=8, spaceAfter=6, fontName=bold))
    for n in ["Normal","Italic","Heading1","Heading2","Heading3","Heading4","Heading5","Heading6"]:
        styles[n].fontName = base

    elems = [
        Paragraph(f"{sn} 투자 분석 보고서", styles["ReportTitle"]),
        Paragraph(f"작성일: {datetime.datetime.now():%Y년 %m월 %d일}", styles["Normal"]),
        Spacer(1,20)
    ]

    color = "green" if state.get("최종_판단")=="투자" else "red"
    elems += [Paragraph(f"최종 판단: <font color='{color}'><b>{state.get('최종_판단','보류')}</b></font>",
                        styles["ReportSubtitle"]), Spacer(1,10)]

    keys   = ["상품_점수","기술_점수","성장률_점수","시장성_점수","경쟁사_점수"]
    labels = ["상품/서비스","기술","성장률","시장성","경쟁사"]
    scores = [int(state.get(k,0)) for k in keys]
    avg    = sum(scores)/len(scores)

    t_data = [["평가 항목","점수"]] + [[l,str(s)] for l,s in zip(labels,scores)] + [["평균",f"{avg:.1f}"]]
    t = Table(t_data, colWidths=[300,100])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(1,0),colors.grey), ("TEXTCOLOR",(0,0),(1,0),colors.whitesmoke),
        ("ALIGN",(0,0),(1,0),"CENTER"), ("FONTNAME",(0,0),(1,0),bold),
        ("FONTNAME",(0,1),(-1,-1),base), ("GRID",(0,0),(-1,-1),1,colors.black),
        ("BACKGROUND",(0,-1),(1,-1),colors.lightgrey)
    ]))
    elems += [
        Paragraph("점수 요약", styles["ReportSubtitle"]),
        t, Spacer(1,12),
        Paragraph(NOTICE, styles["Normal"]),    # 안내문 삽입
        Spacer(1,20)
    ]

    elems.append(Paragraph("상세 분석", styles["ReportSubtitle"]))
    for ln in state.get("보고서","").splitlines():
        if   ln.startswith("### "): elems.append(Paragraph(ln[4:], styles["SectionTitle"]))
        elif ln.startswith("## " ): elems.append(Paragraph(ln[3:], styles["SectionTitle"]))
        elif ln.startswith("# "  ): elems.append(Paragraph(ln[2:], styles["SectionTitle"]))
        elif ln.strip():           elems.append(Paragraph(ln.strip(), styles["Normal"]))
        else:                      elems.append(Spacer(1,6))

    elems += [Spacer(1,30),
              Paragraph("이 보고서는 AI 분석 시스템에 의해 자동 생성되었습니다. "
                        f"© {datetime.datetime.now().year}", styles["Italic"])]
    doc.build(elems)

    # Markdown + wkhtmltopdf
    generate_markdown(state, md_path)
    convert_md_to_pdf(md_path, htmlpdf)

    state.update(pdf_path_reportlab = lab_pdf,
                 pdf_path_wkhtml   = htmlpdf,
                 markdown_path     = md_path,
                 pdf_path          = htmlpdf)
    return state

# %%
# 4. Graph 정의 및 연결
graph = StateGraph(AgentState)

graph.add_node("AnalyzeProduct", analyze_product)
graph.add_node("AnalyzeTechnology", analyze_technology)
graph.add_node("AnalyzeGrowth", analyze_growth)
graph.add_node("InternalJudgement", internal_judgement)
graph.add_node("AnalyzeMarket", analyze_market)
graph.add_node("AnalyzeCompetitor", analyze_competitor)
graph.add_node("FinalJudgement", final_judgement)
graph.add_node("GenerateReport", generate_report)
graph.add_node("GeneratePDF", generate_pdf)  # PDF 생성 노드 추가

graph.set_entry_point("AnalyzeProduct")

graph.add_edge("AnalyzeProduct", "AnalyzeTechnology")
graph.add_edge("AnalyzeTechnology", "AnalyzeGrowth")
graph.add_edge("AnalyzeGrowth", "InternalJudgement")

# 정의한 조건에 따라 다른 노드로 이동
def route_after_internal_judgement(state: AgentState) -> str:
    if state.get("최종_판단") == "보류":
        return "GenerateReport"
    return "AnalyzeMarket"

graph.add_conditional_edges(
    "InternalJudgement",
    route_after_internal_judgement
)

graph.add_edge("AnalyzeMarket", "AnalyzeCompetitor")
graph.add_edge("AnalyzeCompetitor", "FinalJudgement")
graph.add_edge("FinalJudgement", "GenerateReport")
graph.add_edge("GenerateReport", "GeneratePDF")  # 보고서 생성 후 PDF 생성
graph.add_edge("GeneratePDF", END)

# %%
# 5. 성능 비교 전용 함수
def run_performance_comparison():
    """검색 방식들의 성능을 비교하는 전용 함수"""
    print("=" * 60)
    print("🔍 검색 방식 성능 비교 시스템")
    print("=" * 60)
    
    # 테스트용 쿼리
    test_queries = [
        "AI 기술 시장 동향",
        "스타트업 투자 트렌드",
        "인공지능 산업 성장성"
    ]
    
    # PDF 문서 로드
    pdf_dir = os.path.join(os.getcwd(), "data")
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    print(f"📄 로드된 PDF 파일: {len(pdf_files)}개")
    
    all_docs = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        loader = PyMuPDFLoader(pdf_path)
        split_docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100))
        all_docs.extend(split_docs)
    
    print(f"📝 총 문서 청크: {len(all_docs)}개")
    
    # 각 쿼리별로 성능 비교
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*20} 쿼리 {i}: {query} {'='*20}")
        comparison_results = compare_retrieval_methods(all_docs, query)
        
        # 결과 저장 (선택적)
        # 여기에 결과를 파일로 저장하는 코드를 추가할 수 있습니다

def run_query_comparison():
    print("=" * 60)
    print("🔍 검색 방식 성능 비교 (스타트업 이름 기반)")
    print("=" * 60)
    startup_name = input("비교할 스타트업 이름(질의)을 입력하세요: ").strip()
    if not startup_name:
        print("❌ 스타트업 이름을 입력해야 합니다.")
        return

    # PDF 문서 로드
    pdf_dir = os.path.join(os.getcwd(), "data")
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    all_docs = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        loader = PyMuPDFLoader(pdf_path)
        split_docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100))
        all_docs.extend(split_docs)

    query = f"{startup_name} 시장성, 시장 규모, 성장성, 수요 동향, 트렌드"
    results = compare_retrieval_methods(all_docs, query)

    # 각 방식별 검색 문서 내용 출력
    for method, data in results.items():
        print(f"\n{'='*20} {method.upper()} 검색 결과 {'='*20}")
        for i, doc in enumerate(data['docs'], 1):
            print(f"[{i}] (Page: {doc.metadata.get('page', '알 수 없음')})\n{doc.page_content[:300]}...\n")

# %%
# 6. 사용자 입력 받기
def get_startup_name():
    """사용자로부터 분석할 AI 스타트업 이름을 입력받는 함수"""
    print("=" * 60)
    print("🤖 AI 스타트업 투자 평가 시스템")
    print("=" * 60)
    print("분석할 AI 스타트업의 이름을 입력해주세요.")
    print("예시: 업스테이지, 베슬AI, 트웰브랩스, 클로바, 네이버클로바 등")
    print("-" * 60)
    
    while True:
        startup_name = input("스타트업 이름: ").strip()
        if startup_name:
            print(f"\n✅ '{startup_name}' 스타트업 분석을 시작합니다...")
            print("분석에는 몇 분 정도 소요될 수 있습니다.")
            print("-" * 60)
            return startup_name
        else:
            print("❌ 스타트업 이름을 입력해주세요.")

# %%
# 7. 실행
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 AI 스타트업 투자 평가 시스템")
    print("=" * 60)
    print("1. 스타트업 분석 실행")
    print("2. 검색 방식 성능 비교 (샘플 쿼리)")
    print("3. 검색 방식 성능 비교 (스타트업 이름 입력)")
    print("-" * 60)
    
    choice = input("선택하세요 (1, 2, 3): ").strip()
    
    if choice == "2":
        run_performance_comparison()
    elif choice == "3":
        run_query_comparison()
    else:
        startup_name = get_startup_name()
        # Graph 컴파일 및 실행
        compiled_graph = graph.compile()
        initial_state = {"startup_name": startup_name}
        try:
            print("🔄 AI 분석 시스템이 작동 중입니다...")
            result = compiled_graph.invoke(initial_state)
            # 결과 출력
            print("\n" + "=" * 60)
            print("✅ 분석 완료!")
            print("=" * 60)
            print(f"📄 보고서 생성 완료: {result['pdf_path']}")
            print(f"📊 총점: {sum([result.get('상품_점수', 0), result.get('기술_점수', 0), result.get('성장률_점수', 0), result.get('시장성_점수', 0), result.get('경쟁사_점수', 0)]) / 5:.1f}/100")
            print(f"🎯 최종 판단: {result.get('최종_판단', 'N/A')}")
            print("\n--- 보고서 내용 미리보기 ---")
            print(result["보고서"][:500] + "...")
        except Exception as e:
            print(f"\n❌ 분석 중 오류가 발생했습니다: {e}")
            print("API 키 설정을 확인하거나 네트워크 연결을 확인해주세요.")

# %%
# 개발용: 직접 실행 시 결과 확인
# result["보고서"]


