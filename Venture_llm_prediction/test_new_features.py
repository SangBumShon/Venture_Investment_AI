#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
새로운 기능 테스트 스크립트
- JSON OutputParser
- Ensemble Retriever
- Sentence-BERT CrossEncoder Reranker
- Pinecone Vector DB
"""

import os
from dotenv import load_dotenv
from startup_invest_evaluation import (
    setup_environment,
    setup_pinecone,
    setup_embeddings,
    setup_reranker,
    create_ensemble_retriever,
    rerank_documents
)
from models.schemas import (
    ProductAnalysis,
    TechnologyAnalysis,
    GrowthAnalysis,
    MarketAnalysis,
    CompetitorAnalysis,
    FinalJudgement
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

def test_json_output_parser():
    """JSON OutputParser 테스트"""
    print("=== JSON OutputParser 테스트 ===")
    
    # ProductAnalysis 테스트
    output_parser = JsonOutputParser(pydantic_object=ProductAnalysis)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "스타트업 '테스트'의 제품을 평가해주세요. 총점과 분석 근거를 제공하세요."
    )
    chain = prompt | llm | output_parser
    
    try:
        result = chain.invoke({"startup_name": "테스트"})
        print(f"✅ ProductAnalysis 테스트 성공: {result}")
    except Exception as e:
        print(f"❌ ProductAnalysis 테스트 실패: {e}")
    
    # FinalJudgement 테스트
    output_parser = JsonOutputParser(pydantic_object=FinalJudgement)
    prompt = ChatPromptTemplate.from_template(
        "스타트업 '테스트'에 대한 투자 여부를 결정해주세요."
    )
    chain = prompt | llm | output_parser
    
    try:
        result = chain.invoke({"startup_name": "테스트"})
        print(f"✅ FinalJudgement 테스트 성공: {result}")
    except Exception as e:
        print(f"❌ FinalJudgement 테스트 실패: {e}")

def test_embeddings():
    """Embeddings 설정 테스트"""
    print("\n=== Embeddings 설정 테스트 ===")
    
    try:
        embeddings = setup_embeddings()
        if embeddings:
            print("✅ Embeddings 설정 성공")
        else:
            print("❌ Embeddings 설정 실패 - API 키 확인 필요")
    except Exception as e:
        print(f"❌ Embeddings 설정 오류: {e}")

def test_pinecone():
    """Pinecone 설정 테스트"""
    print("\n=== Pinecone 설정 테스트 ===")
    
    try:
        index_name = setup_pinecone()
        if index_name:
            print(f"✅ Pinecone 설정 성공 - 인덱스: {index_name}")
        else:
            print("❌ Pinecone 설정 실패 - API 키 확인 필요")
    except Exception as e:
        print(f"❌ Pinecone 설정 오류: {e}")

def test_reranker():
    """Sentence-BERT CrossEncoder Reranker 테스트"""
    print("\n=== Reranker 설정 테스트 ===")
    
    try:
        reranker = setup_reranker()
        if reranker:
            print("✅ Reranker 설정 성공")
            
            # 간단한 재순위화 테스트
            query = "AI 스타트업 기술"
            documents = [
                Document(page_content="AI 기술을 활용한 스타트업 분석", metadata={"source": "doc1"}),
                Document(page_content="일반적인 기업 경영 정보", metadata={"source": "doc2"}),
                Document(page_content="AI와 머신러닝 기술 동향", metadata={"source": "doc3"})
            ]
            
            reranked_docs = rerank_documents(query, documents, reranker, top_k=2)
            print(f"✅ 문서 재순위화 성공 - {len(reranked_docs)}개 문서")
        else:
            print("❌ Reranker 설정 실패")
    except Exception as e:
        print(f"❌ Reranker 설정 오류: {e}")

def test_ensemble_retriever():
    """Ensemble Retriever 테스트"""
    print("\n=== Ensemble Retriever 테스트 ===")
    
    try:
        # 테스트 문서 생성
        documents = [
            Document(page_content="AI 스타트업의 기술력과 혁신성에 대한 분석", metadata={"source": "doc1"}),
            Document(page_content="스타트업 투자 환경과 시장 동향", metadata={"source": "doc2"}),
            Document(page_content="인공지능 기술의 발전과 응용 분야", metadata={"source": "doc3"}),
            Document(page_content="벤처 캐피탈 투자 전략과 포트폴리오 관리", metadata={"source": "doc4"}),
            Document(page_content="AI 스타트업의 성장 모델과 수익화 전략", metadata={"source": "doc5"})
        ]
        
        embeddings = setup_embeddings()
        reranker = setup_reranker()
        
        if documents and embeddings:
            ensemble_retriever, reranker = create_ensemble_retriever(documents, embeddings, reranker)
            
            if ensemble_retriever:
                # 검색 테스트
                query = "AI 스타트업 기술력"
                retrieved_docs = ensemble_retriever.get_relevant_documents(query)
                print(f"✅ Ensemble Retriever 검색 성공 - {len(retrieved_docs)}개 문서 검색")
                
                # 재순위화 테스트
                if reranker:
                    reranked_docs = rerank_documents(query, retrieved_docs, reranker, top_k=3)
                    print(f"✅ 문서 재순위화 성공 - {len(reranked_docs)}개 문서")
            else:
                print("❌ Ensemble Retriever 생성 실패")
        else:
            print("❌ 문서 또는 embeddings 설정 실패")
    except Exception as e:
        print(f"❌ Ensemble Retriever 테스트 오류: {e}")

def main():
    """메인 테스트 함수"""
    print("🚀 새로운 기능 테스트 시작\n")
    
    # 환경 설정
    setup_environment()
    
    # 각 기능별 테스트
    test_json_output_parser()
    test_embeddings()
    test_pinecone()
    test_reranker()
    test_ensemble_retriever()
    
    print("\n🎉 테스트 완료!")

if __name__ == "__main__":
    main() 