#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Startup Investment Evaluation Agent

This script provides an automated investment evaluation system for AI startups
using LangGraph-based agent workflow with GPT-4 analysis.

Created from original Jupyter notebook
"""

import os
import re
import json
import requests
import datetime
import markdown2
import pdfkit
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain_teddynote import logging
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Vector DB and Retrieval imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import Pinecone
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.tools.tavily_search import TavilySearchResults

# Sentence-BERT CrossEncoder for reranking
from sentence_transformers import CrossEncoder

# PDF generation imports
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Pinecone imports
import pinecone

# Local imports
from models.schemas import (
    ProductAnalysis,
    TechnologyAnalysis,
    GrowthAnalysis,
    MarketAnalysis,
    CompetitorAnalysis,
    FinalJudgement,
    AgentState
)


def setup_environment():
    """
    환경 설정 함수
    - .env 파일에서 API 키 로드
    - LangSmith 로깅 초기화
    - 시스템 환경 설정 완료
    """
    load_dotenv()                    # .env 파일 로드
    logging.langsmith("AI-project")  # LangSmith 로깅 설정
    print("Environment setup complete")


def setup_pinecone():
    """
    Pinecone 벡터 데이터베이스 설정
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "venture-evaluation")
    
    if not pinecone_api_key or not pinecone_environment:
        print("Pinecone API 키 또는 환경이 설정되지 않았습니다.")
        return None
    
    try:
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        
        # 인덱스가 존재하지 않으면 생성
        if pinecone_index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=pinecone_index_name,
                dimension=1536,  # OpenAI embeddings dimension
                metric="cosine"
            )
            print(f"Pinecone 인덱스 '{pinecone_index_name}' 생성됨")
        
        return pinecone_index_name
    except Exception as e:
        print(f"Pinecone 설정 오류: {e}")
        return None


def setup_embeddings():
    """
    OpenAI embeddings 설정
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("OpenAI API 키가 설정되지 않았습니다.")
        return None
    
    return OpenAIEmbeddings(openai_api_key=openai_api_key)


def setup_reranker():
    """
    Sentence-BERT CrossEncoder reranker 설정
    """
    try:
        # 한국어에 최적화된 CrossEncoder 모델 사용
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("Sentence-BERT CrossEncoder reranker 설정 완료")
        return reranker
    except Exception as e:
        print(f"Reranker 설정 오류: {e}")
        return None


def create_ensemble_retriever(documents, embeddings, reranker=None):
    """
    Ensemble retriever 생성 (BM25 + Vector Search)
    """
    try:
        # BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 10
        
        # Vector retriever (Pinecone)
        pinecone_index_name = setup_pinecone()
        if pinecone_index_name and embeddings:
            vectorstore = Pinecone.from_documents(
                documents, 
                embeddings, 
                index_name=pinecone_index_name
            )
            vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        else:
            print("Pinecone 설정이 필요합니다. .env 파일에서 PINECONE_API_KEY와 PINECONE_ENVIRONMENT를 확인해주세요.")
            return None, None
        
        # Ensemble retriever 생성
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.3, 0.7]  # BM25 30%, Vector 70%
        )
        
        print("Ensemble retriever 설정 완료")
        return ensemble_retriever, reranker
        
    except Exception as e:
        print(f"Ensemble retriever 설정 오류: {e}")
        return None, None


def rerank_documents(query, documents, reranker, top_k=5):
    """
    Sentence-BERT CrossEncoder를 사용하여 문서 재순위화
    """
    if not reranker or not documents:
        return documents[:top_k]
    
    try:
        # 쿼리-문서 쌍 생성
        pairs = [[query, doc.page_content] for doc in documents]
        
        # 점수 계산
        scores = reranker.predict(pairs)
        
        # 점수와 문서를 함께 정렬
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 k개 문서 반환
        reranked_docs = [doc for doc, score in doc_score_pairs[:top_k]]
        
        print(f"문서 재순위화 완료: {len(reranked_docs)}개 문서")
        return reranked_docs
        
    except Exception as e:
        print(f"문서 재순위화 오류: {e}")
        return documents[:top_k]


def extract_total_score_from_analysis(analysis: str) -> int:
    patterns = [
        r"\*\*총점\*\*[:：]?\s*(\d{1,3})",
        r"총점[:：]?\s*(\d{1,3})\s*(?:점|/100)?",
        r"Score[:：]?\s*(\d{1,3})\s*(?:점|/100)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, analysis, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None

def extract_checklist_scores(analysis: str, checklist: list) -> int:
    clean_analysis = re.sub(r"총점[:：]?\s*\d{1,3}\s*(?:점|/100)?", "", analysis, flags=re.IGNORECASE)
    total_score = 0
    for i, item in enumerate(checklist, 1):
        patterns = [
            fr"{i}\.\s.*?(\d{{1,2}})\s*/\s*10",
            fr"{i}\.\s.*?(\d{{1,2}})점",
            fr"{i}\.\s.*?점수[:：]?\s*(\d{{1,2}})",
            fr"{item}.*?(\d{{1,2}})\s*/\s*10",
            fr"{item}.*?(\d{{1,2}})점",
            fr"{item}.*?점수[:：]?\s*(\d{{1,2}})",
        ]
        found = False
        for pattern in patterns:
            match = re.search(pattern, clean_analysis, re.DOTALL | re.IGNORECASE)
            if match:
                item_score = int(match.group(1))
                total_score += item_score
                found = True
                break
    return min(100, max(0, total_score))


def extract_json_from_llm_response(text: str) -> str:
    """LLM 응답에서 코드블록(```json ... ```) 또는 여분 텍스트를 제거하고 JSON만 추출"""
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def analyze_product(state: AgentState) -> AgentState:
    """
    상품/서비스 경쟁력 분석 에이전트
    - 스타트업의 제품/서비스가 얼마나 경쟁력이 있는지 평가
    - 외부 API(Tavily, Naver)로 최신 정보 수집
    - 10개 체크리스트 항목으로 종합 평가
    """
    print(f"\n[DEBUG] === 상품 분석 에이전트 시작 ===")
    print(f"[DEBUG] 스타트업 이름: {state.get('startup_name', '없음')}")
    
    startup_name = state.get("startup_name", "")
    if not startup_name:
        print(f"[DEBUG] 스타트업 이름이 없음 - 종료")
        state["상품_점수"] = 0
        state["상품_분석_근거"] = "스타트업 이름이 제공되지 않았습니다."
        return state

    # API 키 확인
    tavily_key = os.getenv("TAVILY_API_KEY")
    naver_id = os.getenv("NAVER_CLIENT_ID")
    naver_secret = os.getenv("NAVER_CLIENT_SECRET")
    
    print(f"[DEBUG] API 키 확인:")
    print(f"[DEBUG] - Tavily API Key: {'설정됨' if tavily_key else '없음'}")
    print(f"[DEBUG] - Naver Client ID: {'설정됨' if naver_id else '없음'}")
    print(f"[DEBUG] - Naver Client Secret: {'설정됨' if naver_secret else '없음'}")
    
    if not tavily_key or not naver_id or not naver_secret:
        print(f"[DEBUG] API 키 누락 - 종료")
        state["상품_점수"] = 0
        state["상품_분석_근거"] = "API 키가 설정되지 않았습니다. .env 파일을 확인해주세요."
        return state

    # 평가 체크리스트 정의
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
    items_formatted = "\n".join(f"{i+1}. {q}" for i, q in enumerate(checklist))

    # Tavily API로 웹 검색
    print(f"[DEBUG] Tavily API 호출 시작...")
    tavily_url = "https://api.tavily.com/v1/search"
    tavily_params = {"query": startup_name, "limit": 5}
    tavily_headers = {"Authorization": f"Bearer {tavily_key}"}
    
    try:
        tavily_res = requests.get(tavily_url, params=tavily_params, headers=tavily_headers)
        print(f"[DEBUG] Tavily API 응답 상태: {tavily_res.status_code}")
        
        if tavily_res.status_code == 200:
            tavily_items = tavily_res.json().get("items", [])
            print(f"[DEBUG] Tavily 검색 결과: {len(tavily_items)}개")
            tavily_context = "\n".join(f"{i+1}. {it.get('title', '제목없음')} ({it.get('url')})" for i, it in enumerate(tavily_items)) or "정보 없음"
        else:
            print(f"[DEBUG] Tavily API 오류: {tavily_res.text}")
            tavily_context = "Tavily API 호출 실패"
    except Exception as e:
        print(f"[DEBUG] Tavily API 예외: {e}")
        tavily_context = f"Tavily API 오류: {e}"

    # Naver News API로 뉴스 검색
    print(f"[DEBUG] Naver API 호출 시작...")
    naver_url = "https://openapi.naver.com/v1/search/news.json"
    naver_headers = {"X-Naver-Client-Id": naver_id, "X-Naver-Client-Secret": naver_secret}
    naver_params = {"query": startup_name, "display": 5}
    
    try:
        naver_res = requests.get(naver_url, params=naver_params, headers=naver_headers)
        print(f"[DEBUG] Naver API 응답 상태: {naver_res.status_code}")
        
        if naver_res.status_code == 200:
            naver_items = naver_res.json().get("items", [])
            print(f"[DEBUG] Naver 검색 결과: {len(naver_items)}개")
            naver_context = "\n".join(f"{i+1}. {it.get('title', '제목없음')} – {it.get('description', '')} ({it.get('originallink')})" for i, it in enumerate(naver_items)) or "정보 없음"
        else:
            print(f"[DEBUG] Naver API 오류: {naver_res.text}")
            naver_context = "Naver API 호출 실패"
    except Exception as e:
        print(f"[DEBUG] Naver API 예외: {e}")
        naver_context = f"Naver API 오류: {e}"

    # Ensemble retriever 및 reranker 설정
    embeddings = setup_embeddings()
    reranker = setup_reranker()
    
    # 문서 로드 및 ensemble retriever 생성
    try:
        # 투자 보고서 문서들 로드
        investment_reports_dir = "investment_reports"
        documents = []
        
        if os.path.exists(investment_reports_dir):
            for filename in os.listdir(investment_reports_dir):
                if filename.endswith(('.pdf', '.md')):
                    file_path = os.path.join(investment_reports_dir, filename)
                    if filename.endswith('.pdf'):
                        loader = PyMuPDFLoader(file_path)
                        documents.extend(loader.load())
                    elif filename.endswith('.md'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            from langchain_core.documents import Document
                            documents.append(Document(page_content=content, metadata={"source": file_path}))
        
        # Ensemble retriever 생성
        if documents and embeddings:
            ensemble_retriever, reranker = create_ensemble_retriever(documents, embeddings, reranker)
            
            # 관련 문서 검색
            query = f"{startup_name} 제품 서비스 기술 특징"
            retrieved_docs = ensemble_retriever.get_relevant_documents(query)
            
            # Reranker로 문서 재순위화
            if reranker:
                retrieved_docs = rerank_documents(query, retrieved_docs, reranker, top_k=5)
            
            # 검색된 문서 내용을 컨텍스트로 변환
            doc_context = "\n".join([f"- {doc.page_content[:500]}..." for doc in retrieved_docs[:3]])
            print(f"[DEBUG] 검색된 문서: {len(retrieved_docs)}개")
        else:
            doc_context = "관련 문서 정보 없음"
            print(f"[DEBUG] 문서 검색 실패")
            
    except Exception as e:
        print(f"[DEBUG] 문서 검색 오류: {e}")
        doc_context = "문서 검색 중 오류 발생"

    # LLM 프롬프트 (JSON OutputParser 사용)
    prompt_template = """당신은 스타트업 '{startup_name}'의 제품/서비스를 다음 체크리스트 10문항에 따라 평가해야 합니다.

체크리스트:
{items}

다음 Tavily API 결과(최대 5개)를 참고하세요:
{tavily_context}

다음 Naver News API 결과(최대 5개)를 참고하세요:
{naver_context}

다음 관련 문서 정보를 참고하세요:
{doc_context}

평가 시 유의사항:
- 정보가 부족하거나 명확하지 않을 경우 관용적으로 5점 내외를 부여할 수 있습니다.
- 부정적 근거가 명확하지 않다면 초기 스타트업 상황을 고려하여 가능성에 가중치를 두고 평가하세요.

점수 부여 기준:
- 매우 우수하거나 충분히 충족 → 9~10점
- 일부 충족되었거나 불완전 → 5~8점
- 거의 정보가 없거나 충족되지 않음 → 0~4점

각 문항별 점수(0~10)와 판단 근거를 분석_근거에 포함하세요.
판단 근거 뒤에는 관련 URL을 괄호 안에 포함하세요. URL이 없을 경우 '정보 없음'으로 표기하세요."""
    
    # JSON OutputParser 설정
    output_parser = JsonOutputParser(pydantic_object=ProductAnalysis)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | output_parser
    
    print(f"[DEBUG] LLM 호출 시작...")
    try:
        result = chain.invoke({
            "startup_name": startup_name,
            "items": items_formatted,
            "tavily_context": tavily_context,
            "naver_context": naver_context,
            "doc_context": doc_context
        })
        
        # JSON OutputParser 결과 처리
        score = result.get('총점', 0)
        reasoning = result.get('분석_근거', '분석 근거 없음')
        print(f"[DEBUG] JSON OutputParser 성공 - 점수: {score}")
        print(f"[DEBUG] 분석 근거 길이: {len(reasoning)}자")
        
    except Exception as e:
        print(f"[DEBUG] LLM 호출 실패: {e}")
        score, reasoning = 0, f"분석 중 오류가 발생했습니다: {e}"

    # Ensure reasoning is a string before slicing
    reasoning_str = str(reasoning) if reasoning is not None else ""
    
    state["상품_점수"] = score
    state["상품_분석_근거"] = reasoning_str
    
    # 노트북 방식처럼 결과 출력
    print(f"\n{'='*50}")
    print(f"상품 분석 결과")
    print(f"{'='*50}")
    print(f"점수: {score}")
    print(f"근거: {reasoning_str}")
    print(f"{'='*50}\n")
    
    return state


def analyze_technology(state: AgentState) -> AgentState:
    """
    기술력 분석 에이전트
    - 스타트업의 기술 수준과 차별성을 평가
    - 특허, 기술 성숙도, 인력 역량 등을 종합 분석
    - 기술적 구현 가능성과 유지보수 용이성 검토
    """
    print(f"\n[DEBUG] === 기술 분석 에이전트 시작 ===")
    print(f"[DEBUG] 스타트업 이름: {state.get('startup_name', '없음')}")
    
    startup_name = state.get("startup_name", "")
    if not startup_name:
        print(f"[DEBUG] 스타트업 이름이 없음 - 종료")
        state["기술_점수"] = 0
        state["기술_분석_근거"] = "스타트업 이름이 제공되지 않았습니다."
        return state

    # API 키 확인
    tavily_key = os.getenv("TAVILY_API_KEY")
    naver_id = os.getenv("NAVER_CLIENT_ID")
    naver_secret = os.getenv("NAVER_CLIENT_SECRET")
    
    print(f"[DEBUG] API 키 확인:")
    print(f"[DEBUG] - Tavily API Key: {'설정됨' if tavily_key else '없음'}")
    print(f"[DEBUG] - Naver Client ID: {'설정됨' if naver_id else '없음'}")
    print(f"[DEBUG] - Naver Client Secret: {'설정됨' if naver_secret else '없음'}")
    
    if not tavily_key or not naver_id or not naver_secret:
        print(f"[DEBUG] API 키 누락 - 종료")
        state["기술_점수"] = 0
        state["기술_분석_근거"] = "API 키가 설정되지 않았습니다. .env 파일을 확인해주세요."
        return state

    # 기술력 평가 체크리스트
    checklist = [
        "기술적 차별성", "특허 보유 여부", "스케일링 가능성", "기술 성숙도",
        "인력 역량", "기술 난이도", "기술 구현 가능성", "기술 유지보수 용이성",
        "기술 표준 준수 여부", "기술 관련 외부 인증 또는 수상 이력"
    ]
    items_formatted = "\n".join(f"{i+1}. {q}" for i, q in enumerate(checklist))

    # Ensemble retriever 및 reranker 설정
    embeddings = setup_embeddings()
    reranker = setup_reranker()
    
    # 문서 로드 및 ensemble retriever 생성
    try:
        # 투자 보고서 문서들 로드
        investment_reports_dir = "investment_reports"
        documents = []
        
        if os.path.exists(investment_reports_dir):
            for filename in os.listdir(investment_reports_dir):
                if filename.endswith(('.pdf', '.md')):
                    file_path = os.path.join(investment_reports_dir, filename)
                    if filename.endswith('.pdf'):
                        loader = PyMuPDFLoader(file_path)
                        documents.extend(loader.load())
                    elif filename.endswith('.md'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            from langchain_core.documents import Document
                            documents.append(Document(page_content=content, metadata={"source": file_path}))
        
        # Ensemble retriever 생성
        if documents and embeddings:
            ensemble_retriever, reranker = create_ensemble_retriever(documents, embeddings, reranker)
            
            # 관련 문서 검색
            query = f"{startup_name} 기술력 특허 스케일링"
            retrieved_docs = ensemble_retriever.get_relevant_documents(query)
            
            # Reranker로 문서 재순위화
            if reranker:
                retrieved_docs = rerank_documents(query, retrieved_docs, reranker, top_k=5)
            
            # 검색된 문서 내용을 컨텍스트로 변환
            doc_context = "\n".join([f"- {doc.page_content[:500]}..." for doc in retrieved_docs[:3]])
            print(f"[DEBUG] 검색된 문서: {len(retrieved_docs)}개")
        else:
            doc_context = "관련 문서 정보 없음"
            print(f"[DEBUG] 문서 검색 실패")
            
    except Exception as e:
        print(f"[DEBUG] 문서 검색 오류: {e}")
        doc_context = "문서 검색 중 오류 발생"

    # Tavily API로 웹 검색
    print(f"[DEBUG] Tavily API 호출 시작...")
    tavily_url = "https://api.tavily.com/v1/search"
    tavily_params = {"query": startup_name, "limit": 5}
    tavily_headers = {"Authorization": f"Bearer {tavily_key}"}
    
    try:
        tavily_res = requests.get(tavily_url, params=tavily_params, headers=tavily_headers)
        print(f"[DEBUG] Tavily API 응답 상태: {tavily_res.status_code}")
        
        if tavily_res.status_code == 200:
            tavily_items = tavily_res.json().get("items", [])
            print(f"[DEBUG] Tavily 검색 결과: {len(tavily_items)}개")
            tavily_context = "\n".join(f"{i+1}. {it.get('title', '제목없음')} ({it.get('url')})" for i, it in enumerate(tavily_items)) or "정보 없음"
        else:
            print(f"[DEBUG] Tavily API 오류: {tavily_res.text}")
            tavily_context = "Tavily API 호출 실패"
    except Exception as e:
        print(f"[DEBUG] Tavily API 예외: {e}")
        tavily_context = f"Tavily API 오류: {e}"

    # Naver API로 뉴스 검색
    print(f"[DEBUG] Naver API 호출 시작...")
    naver_url = "https://openapi.naver.com/v1/search/news.json"
    naver_headers = {"X-Naver-Client-Id": naver_id, "X-Naver-Client-Secret": naver_secret}
    naver_params = {"query": startup_name, "display": 5}
    
    try:
        naver_res = requests.get(naver_url, params=naver_params, headers=naver_headers)
        print(f"[DEBUG] Naver API 응답 상태: {naver_res.status_code}")
        
        if naver_res.status_code == 200:
            naver_items = naver_res.json().get("items", [])
            print(f"[DEBUG] Naver 검색 결과: {len(naver_items)}개")
            naver_context = "\n".join(f"{i+1}. {it.get('title', '제목없음')} – {it.get('description', '')} ({it.get('originallink')})" for i, it in enumerate(naver_items)) or "정보 없음"
        else:
            print(f"[DEBUG] Naver API 오류: {naver_res.text}")
            naver_context = "Naver API 호출 실패"
    except Exception as e:
        print(f"[DEBUG] Naver API 예외: {e}")
        naver_context = f"Naver API 오류: {e}"

    # LLM 프롬프트 (JSON OutputParser 사용)
    prompt_template = """당신은 스타트업 '{startup_name}'의 기술력을 다음 체크리스트 10문항에 따라 평가해야 합니다.

체크리스트:
{items}

다음 Tavily API 결과(최대 5개)를 참고하세요:
{tavily_context}

다음 Naver News API 결과(최대 5개)를 참고하세요:
{naver_context}

다음 관련 문서 정보를 참고하세요:
{doc_context}

평가 시 유의사항:
- 정보가 부족하거나 명확하지 않을 경우 관용적으로 5점 내외를 부여할 수 있습니다.
- 부정적 근거가 명확하지 않다면 초기 스타트업 상황을 고려하여 가능성에 가중치를 두고 평가하세요.

점수 부여 기준:
- 매우 우수하거나 충분히 충족 → 9~10점
- 일부 충족되었거나 불완전 → 5~8점
- 거의 정보가 없거나 충족되지 않음 → 0~4점

각 문항별 점수(0~10)와 판단 근거를 분석_근거에 포함하세요.
판단 근거 뒤에는 관련 URL을 괄호 안에 포함하세요. URL이 없을 경우 '정보 없음'으로 표기하세요."""
    
    # JSON OutputParser 설정
    output_parser = JsonOutputParser(pydantic_object=TechnologyAnalysis)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | output_parser
    
    # LLM 호출
    print(f"[DEBUG] LLM 호출 시작...")
    try:
        result = chain.invoke({
            "startup_name": startup_name,
            "items": items_formatted,
            "tavily_context": tavily_context,
            "naver_context": naver_context,
            "doc_context": doc_context
        })
        
        # JSON OutputParser 결과 처리
        score = result.get('총점', 0)
        reasoning = result.get('분석_근거', '분석 근거 없음')
        print(f"[DEBUG] JSON OutputParser 성공 - 점수: {score}")
        print(f"[DEBUG] 분석 근거 길이: {len(reasoning)}자")
        
    except Exception as e:
        print(f"[DEBUG] LLM 호출 실패: {e}")
        score, reasoning = 0, f"분석 중 오류가 발생했습니다: {e}"

    # Ensure reasoning is a string before slicing
    reasoning_str = str(reasoning) if reasoning is not None else ""
    
    state["기술_점수"] = score
    state["기술_분석_근거"] = reasoning_str
    
    # 노트북 방식처럼 결과 출력
    print(f"\n{'='*50}")
    print(f"기술 분석 결과")
    print(f"{'='*50}")
    print(f"점수: {score}")
    print(f"근거: {reasoning_str}")
    print(f"{'='*50}\n")
    
    return state


def analyze_growth(state: AgentState) -> AgentState:
    """
    성장률 분석 에이전트
    - 스타트업의 성장 가능성과 시장 트렌드 적합성을 평가
    - 매출, 사용자, 투자 유치 등 다양한 성장 지표 분석
    - 해외 진출 및 신시장 확장 속도 검토
    """
    print(f"\n[DEBUG] === 성장률 분석 에이전트 시작 ===")
    print(f"[DEBUG] 스타트업 이름: {state.get('startup_name', '없음')}")
    
    startup_name = state.get("startup_name", "")
    if not startup_name:
        print(f"[DEBUG] 스타트업 이름이 없음 - 종료")
        state["성장률_점수"] = 0
        state["성장률_분석_근거"] = "스타트업 이름이 제공되지 않았습니다."
        return state

    # API 키 확인
    tavily_key = os.getenv("TAVILY_API_KEY")
    naver_id = os.getenv("NAVER_CLIENT_ID")
    naver_secret = os.getenv("NAVER_CLIENT_SECRET")
    
    print(f"[DEBUG] API 키 확인:")
    print(f"[DEBUG] - Tavily API Key: {'설정됨' if tavily_key else '없음'}")
    print(f"[DEBUG] - Naver Client ID: {'설정됨' if naver_id else '없음'}")
    print(f"[DEBUG] - Naver Client Secret: {'설정됨' if naver_secret else '없음'}")
    
    if not tavily_key or not naver_id or not naver_secret:
        print(f"[DEBUG] API 키 누락 - 종료")
        state["성장률_점수"] = 0
        state["성장률_분석_근거"] = "API 키가 설정되지 않았습니다. .env 파일을 확인해주세요."
        return state

    # 성장률 평가 체크리스트
    checklist = [
        "매출 성장률", "사용자 증가율", "시장 점유율 변화", "고객 유지율 (Retention Rate)",
        "월간/분기별 활성 사용자 증가 (MAU/WAU)", "신규 계약/클라이언트 수 증가", "연간 반복 매출(ARR) 성장",
        "투자 유치 규모 변화", "직원 수 증가율", "해외/신시장 진출 속도"
    ]
    items_formatted = "\n".join(f"{i+1}. {q}" for i, q in enumerate(checklist))

    # Ensemble retriever 및 reranker 설정
    embeddings = setup_embeddings()
    reranker = setup_reranker()
    
    # 문서 로드 및 ensemble retriever 생성
    try:
        # 투자 보고서 문서들 로드
        investment_reports_dir = "investment_reports"
        documents = []
        
        if os.path.exists(investment_reports_dir):
            for filename in os.listdir(investment_reports_dir):
                if filename.endswith(('.pdf', '.md')):
                    file_path = os.path.join(investment_reports_dir, filename)
                    if filename.endswith('.pdf'):
                        loader = PyMuPDFLoader(file_path)
                        documents.extend(loader.load())
                    elif filename.endswith('.md'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            from langchain_core.documents import Document
                            documents.append(Document(page_content=content, metadata={"source": file_path}))
        
        # Ensemble retriever 생성
        if documents and embeddings:
            ensemble_retriever, reranker = create_ensemble_retriever(documents, embeddings, reranker)
            
            # 관련 문서 검색
            query = f"{startup_name} 성장률 매출 사용자 증가"
            retrieved_docs = ensemble_retriever.get_relevant_documents(query)
            
            # Reranker로 문서 재순위화
            if reranker:
                retrieved_docs = rerank_documents(query, retrieved_docs, reranker, top_k=5)
            
            # 검색된 문서 내용을 컨텍스트로 변환
            doc_context = "\n".join([f"- {doc.page_content[:500]}..." for doc in retrieved_docs[:3]])
            print(f"[DEBUG] 검색된 문서: {len(retrieved_docs)}개")
        else:
            doc_context = "관련 문서 정보 없음"
            print(f"[DEBUG] 문서 검색 실패")
            
    except Exception as e:
        print(f"[DEBUG] 문서 검색 오류: {e}")
        doc_context = "문서 검색 중 오류 발생"

    # API 호출 - 웹 검색 및 뉴스 검색
    print(f"[DEBUG] Tavily API 호출 시작...")
    tavily_url = "https://api.tavily.com/v1/search"
    tavily_params = {"query": startup_name, "limit": 5}
    tavily_headers = {"Authorization": f"Bearer {tavily_key}"}
    
    try:
        tavily_res = requests.get(tavily_url, params=tavily_params, headers=tavily_headers)
        print(f"[DEBUG] Tavily API 응답 상태: {tavily_res.status_code}")
        
        if tavily_res.status_code == 200:
            tavily_items = tavily_res.json().get("items", [])
            print(f"[DEBUG] Tavily 검색 결과: {len(tavily_items)}개")
            tavily_context = "\n".join(f"{i+1}. {it.get('title', '제목없음')} ({it.get('url')})" for i, it in enumerate(tavily_items)) or "정보 없음"
        else:
            print(f"[DEBUG] Tavily API 오류: {tavily_res.text}")
            tavily_context = "Tavily API 호출 실패"
    except Exception as e:
        print(f"[DEBUG] Tavily API 예외: {e}")
        tavily_context = f"Tavily API 오류: {e}"

    print(f"[DEBUG] Naver API 호출 시작...")
    naver_url = "https://openapi.naver.com/v1/search/news.json"
    naver_headers = {"X-Naver-Client-Id": naver_id, "X-Naver-Client-Secret": naver_secret}
    naver_params = {"query": startup_name, "display": 5}
    
    try:
        naver_res = requests.get(naver_url, params=naver_params, headers=naver_headers)
        print(f"[DEBUG] Naver API 응답 상태: {naver_res.status_code}")
        
        if naver_res.status_code == 200:
            naver_items = naver_res.json().get("items", [])
            print(f"[DEBUG] Naver 검색 결과: {len(naver_items)}개")
            naver_context = "\n".join(f"{i+1}. {it.get('title', '제목없음')} – {it.get('description', '')} ({it.get('originallink')})" for i, it in enumerate(naver_items)) or "정보 없음"
        else:
            print(f"[DEBUG] Naver API 오류: {naver_res.text}")
            naver_context = "Naver API 호출 실패"
    except Exception as e:
        print(f"[DEBUG] Naver API 예외: {e}")
        naver_context = f"Naver API 오류: {e}"

    # LLM 프롬프트 (JSON OutputParser 사용)
    prompt_template = """당신은 스타트업 '{startup_name}'의 성장률을 다음 체크리스트 10문항에 따라 평가해야 합니다.

체크리스트:
{items}

다음 Tavily API 결과(최대 5개)를 참고하세요:
{tavily_context}

다음 Naver News API 결과(최대 5개)를 참고하세요:
{naver_context}

다음 관련 문서 정보를 참고하세요:
{doc_context}

평가 시 유의사항:
- 정보가 부족하거나 명확하지 않을 경우 '정보 부족으로 점수 유보' 대신 관용적으로 5점 내외를 부여할 수 있습니다.
- 부정적 근거가 명확하지 않다면 초기 스타트업 상황을 고려하여 가능성에 가중치를 두고 평가하세요.

점수 부여 기준:
- 매우 우수하거나 충분히 충족 → 9~10점
- 일부 충족되었거나 불완전 → 5~8점
- 거의 정보가 없거나 충족되지 않음 → 0~4점

각 문항별 점수(0~10)와 판단 근거를 분석_근거에 포함하세요.
판단 근거 뒤에는 관련 URL을 괄호 안에 포함하세요. URL이 없을 경우 '정보 없음'으로 표기하세요."""
    
    # JSON OutputParser 설정
    output_parser = JsonOutputParser(pydantic_object=GrowthAnalysis)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | output_parser
    
    # LLM 호출
    print(f"[DEBUG] LLM 호출 시작...")
    try:
        result = chain.invoke({
            "startup_name": startup_name,
            "items": items_formatted,
            "tavily_context": tavily_context,
            "naver_context": naver_context,
            "doc_context": doc_context
        })
        
        # JSON OutputParser 결과 처리
        score = result.get('총점', 0)
        reasoning = result.get('분석_근거', '분석 근거 없음')
        print(f"[DEBUG] JSON OutputParser 성공 - 점수: {score}")
        print(f"[DEBUG] 분석 근거 길이: {len(reasoning)}자")
        
    except Exception as e:
        print(f"[DEBUG] LLM 호출 실패: {e}")
        score, reasoning = 0, f"분석 중 오류가 발생했습니다: {e}"

    # Ensure reasoning is a string before slicing
    reasoning_str = str(reasoning) if reasoning is not None else ""
    
    state["성장률_점수"] = score
    state["성장률_분석_근거"] = reasoning_str
    
    # 노트북 방식처럼 결과 출력
    print(f"\n{'='*50}")
    print(f"성장률 분석 결과")
    print(f"{'='*50}")
    print(f"점수: {score}")
    print(f"근거: {reasoning_str}")
    print(f"{'='*50}\n")
    
    return state


def internal_judgement(state: AgentState) -> AgentState:
    """
    내부 판단 에이전트
    - 상품, 기술, 성장률 점수를 종합하여 1차 판단
    - 하나라도 40점 미만이거나 평균 60점 미만이면 보류 처리
    - 이 단계에서 보류되면 시장성/경쟁사 분석 생략
    """
    print(f"\n[DEBUG] === 내부 판단 에이전트 시작 ===")
    print(f"[DEBUG] 현재 점수:")
    print(f"[DEBUG] - 상품 점수: {state.get('상품_점수', 0)}")
    print(f"[DEBUG] - 기술 점수: {state.get('기술_점수', 0)}")
    print(f"[DEBUG] - 성장률 점수: {state.get('성장률_점수', 0)}")
    
    avg_score = (state.get("상품_점수", 0) + state.get("기술_점수", 0) + state.get("성장률_점수", 0)) / 3
    print(f"[DEBUG] 평균 점수: {avg_score:.1f}")
    
    # 개별 항목 중 하나라도 40점 미만이면 보류
    if (
        state.get("상품_점수", 0) < 40 or
        state.get("기술_점수", 0) < 40 or
        state.get("성장률_점수", 0) < 40
    ):
        print(f"[DEBUG] 개별 항목 40점 미만으로 보류")
        state["최종_판단"] = "보류"
    # 평균 점수가 60점 미만이면 보류
    elif avg_score < 60:
        print(f"[DEBUG] 평균 점수 60점 미만으로 보류")
        state["최종_판단"] = "보류"
    else:
        print(f"[DEBUG] 조건 충족으로 다음 단계 진행")
        state["최종_판단"] = "투자"
    
    print(f"[DEBUG] 최종 판단: {state['최종_판단']}")
    print(f"[DEBUG] === 내부 판단 에이전트 완료 ===")
    return state


def analyze_market(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    시장성 분석 에이전트
    - RAG 기반 시장 분석 (PDF 문서에서 정보 검색)
    - 웹 검색을 통한 최신 시장 동향 파악
    - 시장 규모, 성장성, 진입 가능성 등 10개 항목 평가
    """
    print(f"\n[DEBUG] === 시장성 분석 에이전트 시작 ===")
    print(f"[DEBUG] 스타트업 이름: {state.get('startup_name', '없음')}")
    
    startup_name = state.get("startup_name", "")
    if not startup_name:
        print(f"[DEBUG] 스타트업 이름이 없음 - 종료")
        state["시장성_점수"] = 0
        state["시장성_분석_근거"] = "스타트업 이름이 제공되지 않았습니다."
        return state

    # data 폴더에서 PDF 파일 로드
    pdf_dir = os.path.join(os.getcwd(), "data")
    print(f"[DEBUG] PDF 폴더 경로: {pdf_dir}")
    print(f"[DEBUG] PDF 폴더 존재 여부: {os.path.exists(pdf_dir)}")
    
    if not os.path.exists(pdf_dir):
        print(f"[DEBUG] 데이터 폴더가 없음 - 종료")
        state["시장성_점수"] = 0
        state["시장성_분석_근거"] = "데이터 폴더가 존재하지 않습니다."
        return state
        
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    print(f"[DEBUG] 폴더 내 PDF 파일 수: {len(pdf_files)}")
    print(f"[DEBUG] PDF 파일 목록: {pdf_files}")

    # PDF 문서 로드 및 벡터화
    print(f"[DEBUG] PDF 문서 로드 시작...")
    all_docs = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"[DEBUG] 로드 중: {pdf_file}")
        try:
            loader = PyMuPDFLoader(pdf_path)
            split_docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100))
            print(f"[DEBUG] '{pdf_file}' → {len(split_docs)} 청크 생성")
            all_docs.extend(split_docs)
        except Exception as e:
            print(f"[DEBUG] PDF 로드 실패: {pdf_file} - {e}")
            continue
    
    print(f"[DEBUG] 총 로드된 문서 청크: {len(all_docs)}개")

    # RAG 기반 정보 검색
    print(f"[DEBUG] RAG 검색 시작...")
    if not all_docs:
        print(f"[DEBUG] 로드된 문서가 없음")
        rag_context = "PDF에서 유의미한 정보 없음"
    else:
        print(f"[DEBUG] 벡터 스토어 생성 중...")
        vector_store = Chroma.from_documents(all_docs, OpenAIEmbeddings())
        retriever = vector_store.as_retriever()
        query = f"{startup_name} 시장성, 시장 규모, 성장성, 수요 동향, 트렌드"
        print(f"[DEBUG] 검색 쿼리: {query}")
        retrieved_docs = retriever.invoke(query)
        print(f"[DEBUG] 검색된 문서: {len(retrieved_docs)}개")
        rag_context = "\n\n".join([f"(Page: {doc.metadata.get('page', '알 수 없음')})\n{doc.page_content}" for doc in retrieved_docs]) or "PDF에서 유의미한 정보 없음"
        print(f"[DEBUG] RAG 컨텍스트 길이: {len(rag_context)}자")

    # 웹 검색으로 최신 시장 동향 파악
    print(f"[DEBUG] 웹 검색 시작...")
    try:
        search_tool = TavilySearchResults(k=10)
        search_query = f"{startup_name} AI 스타트업 시장성, 시장 규모, 성장성, 수요 동향, 트렌드 최근 6개월 기사"
        print(f"[DEBUG] 웹 검색 쿼리: {search_query}")
        web_results = search_tool.invoke(search_query)
        print(f"[DEBUG] 웹 검색 결과: {len(web_results)}개")
        web_context = "\n".join([f"{i+1}. {result['title']} ({result['url']})" for i, result in enumerate(web_results)]) or "웹 검색에서 유의미한 정보 없음"
        print(f"[DEBUG] 웹 컨텍스트 길이: {len(web_context)}자")
    except Exception as e:
        print(f"[DEBUG] 웹 검색 실패: {e}")
        web_context = f"웹 검색 실패: {e}"

    # RAG와 웹 검색 결과 통합
    combined_context = f"[PDF 기반 RAG 검색 결과]\n{rag_context}\n\n[웹 검색 결과]\n{web_context}"

    # 시장성 평가 체크리스트
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

    # JSON 형식 응답을 요구하는 프롬프트
    checklist_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(checklist)])
    prompt = ChatPromptTemplate.from_template("""당신은 AI 스타트업 시장성 평가 전문가입니다. '{startup_name}'의 시장성을 종합적으로 평가해 주세요.

다음 정보를 종합하여 분석하세요:
{combined_context}

체크리스트:
""" + checklist_text + """

각 항목은 0점에서 10점 사이로 자유롭게 점수를 부여하세요.

**중요: 반드시 다음 JSON 형식으로 응답해주세요:**
```json
{{
  "총점": 75,
  "분석_근거": "각 항목별 점수와 출처 포함한 분석 근거를 자세히 작성"
}}
```

분석_근거에 포함할 내용:
- 각 항목별 점수(0~10)와 판단 근거
- 출처는 기사 제목, URL 또는 PDF 페이지 번호 명시
- 결론 및 종합 분석""")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm

    print(f"[DEBUG] LLM 호출 시작...")
    try:
        response = chain.invoke({
            "startup_name": startup_name,
            "combined_context": combined_context,
        })
        response_text = response.content
        print(f"[DEBUG] LLM 응답: {response_text[:200]}...")
        json_text = extract_json_from_llm_response(response_text)
        try:
            data = json.loads(json_text)
            score = int(data.get('총점', 0))
            reasoning = data.get('분석_근거', response_text)
            print(f"[DEBUG] JSON 파싱 성공 - 점수: {score}")
        except Exception as e:
            print(f"[DEBUG] json.loads 실패: {e}")
            score = extract_total_score_from_analysis(response_text)
            if score is None:
                score = extract_checklist_scores(response_text, checklist)
            reasoning = response_text
            print(f"[DEBUG] 정규식 추출 - 점수: {score}")
        print(f"[DEBUG] 분석 근거 길이: {len(reasoning)}자")
    except Exception as e:
        print(f"[DEBUG] LLM 호출 실패: {e}")
        score, reasoning = 0, f"분석 중 오류가 발생했습니다: {e}"

    # Ensure reasoning is a string before slicing
    reasoning_str = str(reasoning) if reasoning is not None else ""
    
    state["시장성_점수"] = score
    state["시장성_분석_근거"] = reasoning_str
    
    # 노트북 방식처럼 결과 출력
    print(f"\n{'='*50}")
    print(f"시장성 분석 결과")
    print(f"{'='*50}")
    print(f"점수: {score}")
    print(f"근거: {reasoning_str}")
    print(f"{'='*50}\n")
    
    return state


def analyze_competitor(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    경쟁사 분석 에이전트
    - 웹 검색으로 경쟁사 환경 분석
    - 시장 진입 장벽, 주요 경쟁사, 차별점 등 평가
    - 투자 유치 상황과 성장 속도 비교 분석
    """
    print(f"\n[DEBUG] === 경쟁사 분석 에이전트 시작 ===")
    print(f"[DEBUG] 스타트업 이름: {state.get('startup_name', '없음')}")
    
    startup_name = state.get("startup_name", "")
    if not startup_name:
        print(f"[DEBUG] 스타트업 이름이 없음 - 종료")
        state["경쟁사_점수"] = 0
        state["경쟁사_분석_근거"] = "스타트업 이름이 제공되지 않았습니다."
        return state

    # 웹 검색으로 경쟁사 정보 수집
    print(f"[DEBUG] 웹 검색 시작...")
    try:
        search_tool = TavilySearchResults(k=20)
        search_query = f"{startup_name} 경쟁사 AI 스타트업 시장 분석 최근 6개월 기사"
        print(f"[DEBUG] 검색 쿼리: {search_query}")
        search_results = search_tool.invoke(search_query)
        print(f"[DEBUG] 검색된 기사 수: {len(search_results)}개")
        
        for idx, result in enumerate(search_results, 1):
            print(f"[DEBUG] {idx}. {result['title']}")

        if len(search_results) < 10:
            print("[DEBUG] 경고: 검색된 기사 수가 10개 미만입니다. 정보 부족으로 분석 신뢰도가 낮을 수 있습니다.")
    except Exception as e:
        print(f"[DEBUG] 웹 검색 실패: {e}")
        search_results = []

    # 경쟁사 분석 체크리스트
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

    # JSON 형식 응답을 요구하는 프롬프트
    prompt = ChatPromptTemplate.from_template("""당신은 AI 스타트업 경쟁 분석 전문가입니다. '{startup_name}'의 경쟁사 환경을 종합적으로 평가해 주세요.

최근 6개월 이내에 발행된 20개 이상의 기사를 기반으로, 다음 체크리스트의 각 항목을 평가하세요.
각 항목은 0점에서 10점 사이로 자유롭게 점수를 부여할 수 있습니다. 점수 부여 기준:
- 매우 우수하거나 충분히 충족 → 9~10점
- 일부 충족되었거나 불완전 → 5~8점
- 거의 정보가 없거나 충족되지 않음 → 0~4점

웹 검색 결과:
{search_results}

체크리스트:
1. 시장 진입 장벽 분석
2. 주요 경쟁사 식별
3. 경쟁사 제품/서비스 차별점
4. 시장 점유율 데이터
5. 가격 전략 비교
6. 기술적 우위 분석
7. 타겟 고객층 중복도
8. 성장 속도 및 추세
9. 투자 유치 상황
10. 경쟁사 대응 전략 가능성

**중요: 반드시 다음 JSON 형식으로 응답해주세요:**
```json
{{
  "총점": 75,
  "분석_근거": "각 항목별 점수와 출처 포함한 분석 근거를 자세히 작성"
}}
```

분석_근거에 포함할 내용:
- 각 항목별 점수(0~10)와 판단 근거
- 출처는 기사 제목 또는 URL 명시
- 결론 및 종합 분석""")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm
    
    print(f"[DEBUG] LLM 호출 시작...")
    try:
        response = chain.invoke({
            "startup_name": startup_name,
            "search_results": search_results
        })
        response_text = response.content
        print(f"[DEBUG] LLM 응답: {response_text[:200]}...")
        json_text = extract_json_from_llm_response(response_text)
        try:
            data = json.loads(json_text)
            score = int(data.get('총점', 0))
            reasoning = data.get('분석_근거', response_text)
            print(f"[DEBUG] JSON 파싱 성공 - 점수: {score}")
        except Exception as e:
            print(f"[DEBUG] json.loads 실패: {e}")
            score = extract_total_score_from_analysis(response_text)
            if score is None:
                score = extract_checklist_scores(response_text, checklist)
            reasoning = response_text
            print(f"[DEBUG] 정규식 추출 - 점수: {score}")
        print(f"[DEBUG] 분석 근거 길이: {len(reasoning)}자")
    except Exception as e:
        print(f"[DEBUG] LLM 호출 실패: {e}")
        score, reasoning = 0, f"분석 중 오류가 발생했습니다: {e}"

    # Ensure reasoning is a string before slicing
    reasoning_str = str(reasoning) if reasoning is not None else ""
    
    state["경쟁사_점수"] = score
    state["경쟁사_분석_근거"] = reasoning_str
    
    # 노트북 방식처럼 결과 출력
    print(f"\n{'='*50}")
    print(f"경쟁사 분석 결과")
    print(f"{'='*50}")
    print(f"점수: {score}")
    print(f"근거: {reasoning_str}")
    print(f"{'='*50}\n")
    
    return state


def final_judgement(state: AgentState) -> AgentState:
    """
    최종 투자 판단 에이전트
    - 모든 분석 결과를 종합하여 최종 투자 여부 결정
    - LLM을 사용하여 종합적인 판단 수행
    """
    print(f"\n[DEBUG] === 최종 판단 에이전트 시작 ===")
    print(f"[DEBUG] 현재 점수:")
    print(f"[DEBUG] - 상품 점수: {state.get('상품_점수', 0)}")
    print(f"[DEBUG] - 기술 점수: {state.get('기술_점수', 0)}")
    print(f"[DEBUG] - 성장률 점수: {state.get('성장률_점수', 0)}")
    print(f"[DEBUG] - 시장성 점수: {state.get('시장성_점수', 0)}")
    print(f"[DEBUG] - 경쟁사 점수: {state.get('경쟁사_점수', 0)}")
    
    startup_name = state.get("startup_name", "알 수 없는 스타트업")
    
    # LLM 프롬프트 (JSON OutputParser 사용)
    prompt_template = """스타트업 '{startup_name}'에 대한 투자 심사 결과를 종합하여 최종 판단을 내려야 합니다.

현재 점수:
- 상품/서비스: {상품_점수}점
- 기술: {기술_점수}점
- 성장률: {성장률_점수}점
- 시장성: {시장성_점수}점
- 경쟁사: {경쟁사_점수}점

각 항목별 분석 근거:
1. 상품/서비스 분석: {상품_분석_근거}
2. 기술 분석: {기술_분석_근거}
3. 성장률 분석: {성장률_분석_근거}
4. 시장성 분석: {시장성_분석_근거}
5. 경쟁사 분석: {경쟁사_분석_근거}

위 분석 결과를 종합하여 투자 여부를 결정하세요. 각 항목의 점수와 분석 근거를 고려하여 종합적인 판단을 내리고, 그 근거를 명확히 제시하세요.

판단 기준:
- 투자: 전반적으로 우수한 성과와 잠재력을 보이는 경우
- 보류: 개선이 필요하거나 위험 요소가 있는 경우"""
    
    # JSON OutputParser 설정
    output_parser = JsonOutputParser(pydantic_object=FinalJudgement)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | output_parser
    
    # LLM 호출
    print(f"[DEBUG] LLM 호출 시작...")
    try:
        result = chain.invoke({
            "startup_name": startup_name,
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
        
        # JSON OutputParser 결과 처리
        final_decision = result.get('최종_판단', '보류')
        decision_reasoning = result.get('판단_근거', '판단 근거 없음')
        
        print(f"[DEBUG] JSON OutputParser 성공 - 최종 판단: {final_decision}")
        print(f"[DEBUG] 판단 근거 길이: {len(decision_reasoning)}자")
        
        state["최종_판단"] = final_decision
        
    except Exception as e:
        print(f"[DEBUG] LLM 호출 실패: {e}")
        # 기본 로직으로 폴백
        avg_internal = (state.get("상품_점수", 0) + state.get("기술_점수", 0) + state.get("성장률_점수", 0)) / 3
        avg_total = (avg_internal + state.get("시장성_점수", 0) + state.get("경쟁사_점수", 0)) / 3
        
        if avg_total >= 65:
            state["최종_판단"] = "투자"
        else:
            state["최종_판단"] = "보류"
    
    print(f"[DEBUG] 최종 판단: {state['최종_판단']}")
    print(f"[DEBUG] === 최종 판단 에이전트 완료 ===")
    return state


def generate_report(state: AgentState) -> AgentState:
    """
    종합 투자 보고서 생성 에이전트
    - 모든 분석 결과를 종합하여 최종 투자 보고서 작성
    - 각 항목별 강점과 약점을 요약
    """
    print(f"\n[DEBUG] === 보고서 생성 에이전트 시작 ===")
    print(f"[DEBUG] 스타트업 이름: {state.get('startup_name', '없음')}")
    print(f"[DEBUG] 최종 판단: {state.get('최종_판단', '없음')}")
    
    # 최종 판단 근거와 개선 제안 포함
    startup_name = state.get("startup_name", "알 수 없는 스타트업")

    # 보고서 생성 프롬프트
    prompt = ChatPromptTemplate.from_template("""스타트업 '{startup_name}'에 대한 투자 심사 결과는 {최종_판단} 입니다. 

점수 요약:
- 상품/서비스: {상품_점수}
- 기술: {기술_점수}
- 성장률: {성장률_점수}
- 시장성: {시장성_점수}
- 경쟁사: {경쟁사_점수}

각 항목별 분석 근거:
1. 상품/서비스 분석:
{상품_분석_근거}

2. 기술 분석:
{기술_분석_근거}

3. 성장률 분석:
{성장률_분석_근거}

4. 시장성 분석:
{시장성_분석_근거}

5. 경쟁사 분석:
{경쟁사_분석_근거}

위 분석 결과를 바탕으로 투자 심사 보고서를 작성하세요. 각 항목별 강점과 약점을 요약하고, 최종 판단의 근거를 명확히 제시하며, 개선이 필요한 부분에 대한 제안도 포함해주세요.""")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm
    
    # 보고서 생성
    print(f"[DEBUG] LLM 호출 시작...")
    try:
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
        
        print(f"[DEBUG] LLM 응답 성공 - 보고서 길이: {len(report.content)}자")
        print(f"[DEBUG] 보고서 미리보기: {report.content[:200]}...")
    except Exception as e:
        print(f"[DEBUG] LLM 호출 실패: {e}")
        report_content = f"보고서 생성 실패: {e}"
        report = type('obj', (object,), {'content': report_content})()
    
    print(f"[DEBUG] === 보고서 생성 에이전트 완료 ===")
    
    state["보고서"] = report.content
    return state


def generate_markdown(state: dict, md_path: str) -> None:
    """
    마크다운 보고서 생성 함수
    - 분석 결과를 마크다운 형식으로 변환
    - 테이블과 스타일링 포함
    """
    sn = state.get("startup_name", "알 수 없는 스타트업")
    rep = state.get("보고서", "보고서 내용이 없습니다.")
    dec = state.get("최종_판단", "보류")

    labels = ["상품/서비스", "기술", "성장률", "시장성", "경쟁사"]
    keys = ["상품_점수", "기술_점수", "성장률_점수", "시장성_점수", "경쟁사_점수"]
    scores = [int(state.get(k, 0)) for k in keys]
    avg = sum(scores) / len(scores)

    dec_html = ("<span style='color:green;'>투자</span>"
                if dec == "투자" else "<span style='color:red;'>보류</span>")

    NOTICE = "상품/서비스, 기술, 성장률 중에 하나라도 40점 미만이거나 평균 60점 미만이면 시장성과 경쟁사 항목은 측정되지 않습니다."

    md = [
        f"<h1 align='center'>{sn} 투자 분석 보고서</h1>", "",
        f"- **작성일** : {datetime.datetime.now():%Y년 %m월 %d일}",
        f"- **최종 판단** : {dec_html}", "", "---", "",
        "## 1. 점수 요약", "",
        "| 평가 항목 | 점수 |", "|:-----------:|:----:|",
        *[f"| {l} | **{v}** |" for l, v in zip(labels, scores)],
        f"| **평균** | **{avg:.1f}** |", "",
        f"> **{NOTICE}**",
        "", "---", "",
        "## 2. 상세 분석", ""
    ]

    # Add detailed analysis
    for line in rep.strip().splitlines():
        if line.startswith("### "):
            md.append(f"#### **{line[4:]}**")
        elif line.startswith("## "):
            md.append(f"### **{line[3:]}**")
        elif line.startswith("# "):
            md.append(f"## **{line[2:]}**")
        else:
            md.append(line)

    md += ["", "---", "*이 보고서는 AI 분석 시스템에 의해 자동 생성되었습니다.*"]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def convert_md_to_pdf(md_path: str, pdf_path: str) -> None:
    """
    마크다운을 PDF로 변환하는 함수
    - wkhtmltopdf를 사용하여 고품질 PDF 생성
    - 한국어 폰트 지원 및 스타일링 적용
    """
    BASE_DIR = os.getcwd()
    WKHTMLTOPDF_BIN = os.path.join(BASE_DIR, "wkhtmltopdf", "bin", "wkhtmltopdf.exe")
    NANUM_REG_TTF = os.path.join(BASE_DIR, "NanumGothic.ttf")
    NANUM_BOLD_TTF = os.path.join(BASE_DIR, "NanumGothicBold.ttf")
    
    try:
        cfg = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_BIN)
        with open(md_path, "r", encoding="utf-8") as f:
            body_html = markdown2.markdown(f.read(), extras=["tables"])

        if os.path.isfile(NANUM_REG_TTF):
            reg = os.path.abspath(NANUM_REG_TTF).replace("\\", "/")
            bold = os.path.abspath(NANUM_BOLD_TTF if os.path.isfile(NANUM_BOLD_TTF) else NANUM_REG_TTF).replace("\\", "/")
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
                           options={"enable-local-file-access": None, "encoding": "utf-8"})
    except Exception as e:
        print(f"PDF 변환 중 오류 발생: {e}")


def generate_pdf(state: dict) -> dict:
    """
    PDF 보고서 생성 에이전트
    - ReportLab과 wkhtmltopdf 두 가지 방식으로 PDF 생성
    - 한국어 폰트 지원 및 전문적인 보고서 레이아웃
    - 마크다운 및 HTML 형식 모두 지원
    """
    sn = state.get("startup_name", "알 수 없는 스타트업")
    today = datetime.datetime.now().strftime("%Y%m%d")
    out = "investment_reports"
    os.makedirs(out, exist_ok=True)
    
    BASE_DIR = os.getcwd()
    NANUM_REG_TTF = os.path.join(BASE_DIR, "NanumGothic.ttf")
    NANUM_BOLD_TTF = os.path.join(BASE_DIR, "NanumGothicBold.ttf")
    
    lab_pdf = os.path.join(out, f"{sn}_투자분석보고서_{today}_lab.pdf")
    md_path = os.path.join(out, f"{sn}_보고서.md")
    htmlpdf = os.path.join(out, f"{sn}_투자분석보고서_{today}.pdf")

    # ReportLab PDF generation
    doc = SimpleDocTemplate(lab_pdf, pagesize=letter)
    styles = getSampleStyleSheet()
    
    try:
        pdfmetrics.registerFont(TTFont("NanumGothic", NANUM_REG_TTF))
        pdfmetrics.registerFont(TTFont("NanumGothic-Bold", NANUM_BOLD_TTF))
        base, bold = "NanumGothic", "NanumGothic-Bold"
    except Exception:
        base, bold = "Helvetica", "Helvetica-Bold"

    styles.add(ParagraphStyle("ReportTitle", parent=styles["Heading1"], fontSize=18, alignment=1, spaceAfter=20, fontName=base))
    styles.add(ParagraphStyle("ReportSubtitle", parent=styles["Heading2"], fontSize=14, spaceBefore=10, spaceAfter=10, fontName=base))
    styles.add(ParagraphStyle("SectionTitle", parent=styles["Normal"], fontSize=12, spaceBefore=8, spaceAfter=6, fontName=bold))
    
    for n in ["Normal", "Italic", "Heading1", "Heading2", "Heading3", "Heading4", "Heading5", "Heading6"]:
        styles[n].fontName = base

    elems = [
        Paragraph(f"{sn} 투자 분석 보고서", styles["ReportTitle"]),
        Paragraph(f"작성일: {datetime.datetime.now():%Y년 %m월 %d일}", styles["Normal"]),
        Spacer(1, 20)
    ]

    color = "green" if state.get("최종_판단") == "투자" else "red"
    elems += [Paragraph(f"최종 판단: <font color='{color}'><b>{state.get('최종_판단', '보류')}</b></font>",
                        styles["ReportSubtitle"]), Spacer(1, 10)]

    keys = ["상품_점수", "기술_점수", "성장률_점수", "시장성_점수", "경쟁사_점수"]
    labels = ["상품/서비스", "기술", "성장률", "시장성", "경쟁사"]
    scores = [int(state.get(k, 0)) for k in keys]
    avg = sum(scores) / len(scores)

    t_data = [["평가 항목", "점수"]] + [[l, str(s)] for l, s in zip(labels, scores)] + [["평균", f"{avg:.1f}"]]
    t = Table(t_data, colWidths=[300, 100])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (1, 0), colors.grey), ("TEXTCOLOR", (0, 0), (1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (1, 0), "CENTER"), ("FONTNAME", (0, 0), (1, 0), bold),
        ("FONTNAME", (0, 1), (-1, -1), base), ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, -1), (1, -1), colors.lightgrey)
    ]))
    
    NOTICE = "상품/서비스, 기술, 성장률 중에 하나라도 40점 미만이거나 평균 60점 미만이면 시장성과 경쟁사 항목은 측정되지 않습니다."
    
    elems += [
        Paragraph("점수 요약", styles["ReportSubtitle"]),
        t, Spacer(1, 12),
        Paragraph(NOTICE, styles["Normal"]),
        Spacer(1, 20)
    ]

    elems.append(Paragraph("상세 분석", styles["ReportSubtitle"]))
    for ln in state.get("보고서", "").splitlines():
        if ln.startswith("### "):
            elems.append(Paragraph(ln[4:], styles["SectionTitle"]))
        elif ln.startswith("## "):
            elems.append(Paragraph(ln[3:], styles["SectionTitle"]))
        elif ln.startswith("# "):
            elems.append(Paragraph(ln[2:], styles["SectionTitle"]))
        elif ln.strip():
            elems.append(Paragraph(ln.strip(), styles["Normal"]))
        else:
            elems.append(Spacer(1, 6))

    elems += [Spacer(1, 30),
              Paragraph("이 보고서는 AI 분석 시스템에 의해 자동 생성되었습니다. "
                        f"© {datetime.datetime.now().year}", styles["Italic"])]
    doc.build(elems)

    # Markdown + wkhtmltopdf
    generate_markdown(state, md_path)
    convert_md_to_pdf(md_path, htmlpdf)

    state.update(
        pdf_path_reportlab=lab_pdf,
        pdf_path_wkhtml=htmlpdf,
        markdown_path=md_path,
        pdf_path=htmlpdf
    )
    return state


def route_after_internal_judgement(state: AgentState) -> str:
    """
    내부 판단 후 라우팅 함수
    - 내부 판단 결과에 따라 다음 단계 결정
    - 보류 시 바로 보고서 생성, 통과 시 시장성 분석 진행
    """
    if state.get("최종_판단") == "보류":
        return "GenerateReport"
    return "AnalyzeMarket"


def create_workflow():
    """
    LangGraph 워크플로우 생성 및 컴파일
    - 9개 에이전트를 순차적으로 연결
    - 조건부 라우팅으로 효율적인 분석 진행
    - 상태 기반 데이터 공유 및 처리
    """
    # 상태 그래프 생성
    graph = StateGraph(AgentState)

    # 에이전트 노드 추가
    graph.add_node("AnalyzeProduct", analyze_product)      # 상품 분석
    graph.add_node("AnalyzeTechnology", analyze_technology) # 기술 분석
    graph.add_node("AnalyzeGrowth", analyze_growth)        # 성장률 분석
    graph.add_node("InternalJudgement", internal_judgement) # 내부 판단
    graph.add_node("AnalyzeMarket", analyze_market)        # 시장성 분석
    graph.add_node("AnalyzeCompetitor", analyze_competitor) # 경쟁사 분석
    graph.add_node("FinalJudgement", final_judgement)      # 최종 판단
    graph.add_node("GenerateReport", generate_report)      # 보고서 생성
    graph.add_node("GeneratePDF", generate_pdf)           # PDF 생성

    # 시작점 설정
    graph.set_entry_point("AnalyzeProduct")

    # 순차적 연결 (상품 → 기술 → 성장률 → 내부판단)
    graph.add_edge("AnalyzeProduct", "AnalyzeTechnology")
    graph.add_edge("AnalyzeTechnology", "AnalyzeGrowth")
    graph.add_edge("AnalyzeGrowth", "InternalJudgement")

    # 조건부 라우팅 (내부판단 결과에 따라 분기)
    graph.add_conditional_edges(
        "InternalJudgement",
        route_after_internal_judgement
    )

    # 외부 분석 단계 (시장성 → 경쟁사 → 최종판단)
    graph.add_edge("AnalyzeMarket", "AnalyzeCompetitor")
    graph.add_edge("AnalyzeCompetitor", "FinalJudgement")
    
    # 보고서 생성 단계 (보고서 → PDF)
    graph.add_edge("FinalJudgement", "GenerateReport")
    graph.add_edge("GenerateReport", "GeneratePDF")
    graph.add_edge("GeneratePDF", END)

    return graph.compile()


def main():
    """
    메인 실행 함수
    - 전체 AI 스타트업 투자 분석 시스템의 진입점
    - 사용자 입력 처리 및 워크플로우 실행
    - 결과 출력 및 에러 처리
    """
    print(f"[DEBUG] === 메인 함수 시작 ===")
    
    # 환경 설정 (API 키, 로깅 등)
    setup_environment()
    
    # 워크플로우 생성 및 컴파일
    print(f"[DEBUG] 워크플로우 생성 중...")
    compiled_graph = create_workflow()
    print(f"[DEBUG] 워크플로우 생성 완료")
    
    # 사용자로부터 스타트업 이름 입력받기
    startup_name = input("분석할 스타트업 이름을 입력하세요 (기본값: 업스테이지): ").strip()
    if not startup_name:
        startup_name = "업스테이지"
    
    print(f"[DEBUG] 분석 대상: {startup_name}")
    
    # 워크플로우 실행
    print(f"\n스타트업 '{startup_name}' 분석을 시작합니다...")
    initial_state = {"startup_name": startup_name}
    
    try:
        # 전체 분석 프로세스 실행
        print(f"[DEBUG] 워크플로우 실행 시작...")
        result = compiled_graph.invoke(initial_state)
        print(f"[DEBUG] 워크플로우 실행 완료")
        
        # 결과 출력
        print(f"\n{'='*60}")
        print(f"보고서 생성 완료!")
        print(f"{'='*60}")
        print(f"스타트업: {result.get('startup_name', '알 수 없음')}")
        print(f"최종 판단: {result.get('최종_판단', '보류')}")
        print(f"PDF 경로: {result.get('pdf_path', '없음')}")
        
        # 점수 요약 출력
        print(f"\n점수 요약:")
        print(f"- 상품/서비스: {result.get('상품_점수', 0)}점")
        print(f"- 기술: {result.get('기술_점수', 0)}점")
        print(f"- 성장률: {result.get('성장률_점수', 0)}점")
        print(f"- 시장성: {result.get('시장성_점수', 0)}점")
        print(f"- 경쟁사: {result.get('경쟁사_점수', 0)}점")
        
        # 평균 점수 계산 및 출력
        avg_score = (result.get('상품_점수', 0) + result.get('기술_점수', 0) + 
                    result.get('성장률_점수', 0) + result.get('시장성_점수', 0) + 
                    result.get('경쟁사_점수', 0)) / 5
        print(f"- 평균: {avg_score:.1f}점")
        
        # 보고서 미리보기 출력
        print(f"\n--- 보고서 미리보기 ---")
        print(result.get("보고서", "보고서 없음")[:500] + "...")
        
    except Exception as e:
        # 에러 처리 및 디버깅 정보 출력
        print(f"[DEBUG] 메인 함수에서 예외 발생: {e}")
        print(f"분석 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"[DEBUG] === 메인 함수 완료 ===")


if __name__ == "__main__":
    main() 