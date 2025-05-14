# AI Startup Investment Evaluation Agent
본 프로젝트는 인공지능 스타트업에 대한 투자 가능성을 자동으로 평가하는 에이전트를 설계하고 구현한 실습 프로젝트입니다.

## Overview
- Objective: AI 스타트업의 상품 경쟁력, 기술력, 시장성, 성장 가능성, 경쟁사 분석 등을 종합적으로 평가하여 투자 적합성을 판단

- Method: LangGraph 기반의 AI 에이전트 흐름 구성 + LangChain + GPT 기반 평가 자동화

- Tools: LangGraph, LangChain, OpenAI GPT-4o, ReportLab, Python

## Features
- 기업 내 외부 항목 구분 평가 
    - 내부 항목 : 상품, 기술, 성장성
    - 외부 항목 : 시장성, 경쟁사

- LLM 기반 항목별 자동 평가
    - 신뢰도 향상을 위해 평가 점수와 근거 제시

- 평가 결과 자동 종합 및 투자 판단 (투자 / 보류)

- 평가 결과를 기반으로 PDF 보고서 자동 생성

- Agent 기반 구성으로 항목별 모듈화 및 확장 가능

## Tech Stack 

| Category   | Details                             |
|------------|-------------------------------------|
| Framework  | LangGraph, LangChain, Python        |
| LLM        | GPT-4o via OpenAI API               |
| PDF Report | ReportLab                           |
| State Mgmt | TypedDict 기반 상태 공유 (AgentState)|
| Logging	 | LangSmith (langchain_teddynote)     |
| VectorDB	 | ChromaDB                            |

## Agents
- Agent `Product`: 상품 경쟁력을 평가함
- Agent `Technology`: 기술 수준과 차별성을 평가함
- Agent `Growth`: 성장 가능성과 시장 트렌드 적합성을 평가함
- Agent `InternalJudgement`: 상품, 기술, 성장성을 종합하여 내부 기준을 평가함
- Agent `Market`: 시장성(시장 수요, 산업 전망 등)을 평가함
- Agent `Competition`: 경쟁사와의 비교를 통해 차별성을 평가함
- Agent `FinalJudgement` : 최종 투자 여부를 판단함
- Agent `GenerateReport`: 평가 결과를 종합하여 보고서를 생성함
- Agent `GeneratePDF`: 보고서를 pdf 형식으로 변환함

## Architecture
![Architecture](images/Architecture2.png)

## LangSmith Tracking Example
![Architecture](images/langgraph.png)

## Directory Structure
```
├── data/                                 # 벡터 DB 활용 문서
├── investment_reports/                   # 평가 보고서 저장
├── startup_invest_evaluation.ipynb       # 실행 스크립트
└── README.md
```

## Contributors 
- 김상헌 : 결론 도출 및 PDF 출력 에이전트
- 김지현 : 경쟁사 분석 에이전트, Vector DB 구축
- 손상범 : 기업 자체 평가 에이전트
- 이상현 : 기업 자체 평가 에이전트
- 이소희 : 시장 분석 에이전트
