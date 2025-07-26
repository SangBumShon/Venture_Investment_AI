import os
import re
import requests
from typing import List, Dict, Any, Optional

def get_api_keys() -> Dict[str, Optional[str]]:
    """API 키들을 환경변수에서 가져오는 함수"""
    return {
        "tavily_key": os.getenv("TAVILY_API_KEY"),
        "naver_id": os.getenv("NAVER_CLIENT_ID"),
        "naver_secret": os.getenv("NAVER_CLIENT_SECRET"),
        "pinecone_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_env": os.getenv("PINECONE_ENV"),
        "pinecone_index": os.getenv("PINECONE_INDEX")
    }

def validate_api_keys(required_keys: List[str]) -> Dict[str, str]:
    """필요한 API 키들이 설정되어 있는지 확인하는 함수"""
    keys = get_api_keys()
    missing_keys = [key for key in required_keys if not keys.get(key)]
    
    if missing_keys:
        raise ValueError(f"다음 API 키들이 설정되지 않았습니다: {', '.join(missing_keys)}. .env 파일을 확인해주세요.")
    
    return {k: v for k, v in keys.items() if k in required_keys}

def search_tavily(query: str, limit: int = 5) -> str:
    """Tavily API를 사용한 웹 검색 함수"""
    keys = validate_api_keys(["tavily_key"])
    
    # Tavily API 최적화된 방식 (성공한 방식만 사용)
    tavily_url = "https://api.tavily.com/search"
    
    # 재시도 로직 추가
    max_retries = 3
    for attempt in range(max_retries):
        try:
            tavily_data = {"query": query, "limit": limit}
            tavily_headers = {"Authorization": f"Bearer {keys['tavily_key']}", "Content-Type": "application/json"}
            
            # print(f"Tavily API 요청 시도 {attempt + 1}/{max_retries}")
            tavily_res = requests.post(tavily_url, json=tavily_data, headers=tavily_headers, timeout=30)
            
            if tavily_res.status_code == 200:
                response_data = tavily_res.json()
                
                # 디버깅용: 전체 JSON 구조 출력 (주석처리)
                # print(f"\n🔧 [DEBUG] Tavily API 전체 응답 구조:")
                # print(f"응답 키들: {list(response_data.keys())}")
                # print(f"결과 개수: {len(response_data.get('results', []))}")
                
                tavily_items = response_data.get("results", [])
                
                if tavily_items:
                    # 디버깅용: Tavily 검색 결과 출력 (주석처리)
                    # print(f"\n🔍 [DEBUG] Tavily 검색 결과 ({len(tavily_items)}개)")
                    # print("=" * 50)
                    # for i, item in enumerate(tavily_items, 1):
                    #     print(f"{i}. {item.get('title', '제목없음')}")
                    #     print(f"   📍 {item.get('url', 'URL 없음')}")
                    #     if item.get('content'):
                    #         content_preview = item.get('content', '')[:150].replace('\n', ' ').strip()
                    #         print(f"   📄 {content_preview}...")
                    #     
                    #     # 디버깅용: 각 아이템의 JSON 키들 출력
                    #     # print(f"   🔧 JSON 키들: {list(item.keys())}")
                    #     print()
                    
                    tavily_context = "\n".join(f"{i+1}. {it.get('title', '제목없음')} ({it.get('url', 'URL 없음')})" for i, it in enumerate(tavily_items))
                    return tavily_context or "정보 없음"
            else:
                print(f"Tavily API 오류: {tavily_res.status_code} - {tavily_res.text}")
                
        except requests.exceptions.Timeout:
            print(f"Tavily API 타임아웃 (시도 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)  # 2초 대기 후 재시도
                continue
        except Exception as e:
            print(f"Tavily API 요청 오류 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)  # 2초 대기 후 재시도
                continue
    
    print("모든 Tavily API 재시도 실패")
    
    return "정보 없음"

def search_naver_news(query: str, display: int = 5) -> str:
    """Naver News API를 사용한 뉴스 검색 함수"""
    keys = validate_api_keys(["naver_id", "naver_secret"])
    
    naver_url = "https://openapi.naver.com/v1/search/news.json"
    naver_headers = {"X-Naver-Client-Id": keys["naver_id"], "X-Naver-Client-Secret": keys["naver_secret"]}
    naver_params = {"query": query, "display": display}
    
    try:
        naver_res = requests.get(naver_url, params=naver_params, headers=naver_headers)
        naver_res.raise_for_status()
        naver_response_data = naver_res.json()
        
        # 디버깅용: 전체 JSON 구조 출력 (주석처리)
        # print(f"\n🔧 [DEBUG] Naver News API 전체 응답 구조:")
        # print(f"응답 키들: {list(naver_response_data.keys())}")
        # print(f"결과 개수: {len(naver_response_data.get('items', []))}")
        
        naver_items = naver_response_data.get("items", [])
        
        if naver_items:
            # 디버깅용: Naver News 검색 결과 출력 (주석처리)
            # print(f"\n📰 [DEBUG] Naver News 검색 결과 ({len(naver_items)}개)")
            # print("=" * 50)
            # for i, item in enumerate(naver_items, 1):
            #     print(f"{i}. {item.get('title', '제목없음')}")
            #     print(f"   📍 {item.get('originallink', 'URL 없음')}")
            #     if item.get('description'):
            #         desc_preview = item.get('description', '')[:150].replace('\n', ' ').strip()
            #         print(f"   📄 {desc_preview}...")
            #     
            #     # 디버깅용: 각 아이템의 JSON 키들 출력
            #     # print(f"   🔧 JSON 키들: {list(item.keys())}")
            #     print()
            pass
        
        naver_context = "\n".join(f"{i+1}. {it.get('title', '제목없음')} – {it.get('description', '')} ({it.get('originallink')})" for i, it in enumerate(naver_items))
        return naver_context or "정보 없음"
    except Exception as e:
        print(f"Naver News API 오류: {e}")
        return "정보 없음"

def extract_total_score_from_analysis(analysis: str) -> Optional[int]:
    """분석 결과에서 총점을 추출하는 함수"""
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

def extract_checklist_scores(analysis: str, checklist: List[str]) -> int:
    """분석 결과에서 체크리스트 항목별 점수를 추출하여 총점을 계산하는 함수"""
    clean_analysis = re.sub(r"총점[:：]?\s*\d{1,3}\s*(?:점|/100)?", "", analysis, flags=re.IGNORECASE)
    total_score = 0
    
    for i, item in enumerate(checklist, 1):
        patterns = [
            fr"{i}\.\s.*?(\d{{1,2}})\s*/\s*10",
            fr"{i}\.\s.*?(\d{{1,2}})점",
            fr"{i}\.\s.*?점수[:：]?\s*(\d{{1,2}})",
            fr"{re.escape(item)}.*?(\d{{1,2}})\s*/\s*10",
            fr"{re.escape(item)}.*?(\d{{1,2}})점",
            fr"{re.escape(item)}.*?점수[:：]?\s*(\d{{1,2}})",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clean_analysis, re.DOTALL | re.IGNORECASE)
            if match:
                item_score = int(match.group(1))
                total_score += item_score
                break
    
    return min(100, max(0, total_score))

def extract_insufficient_items(text: str, checklist: List[str]) -> List[str]:
    """분석 결과에서 정보가 부족한 항목들을 추출하는 함수"""
    return [
        item for item in checklist
        if re.search(fr"{re.escape(item)}.*?0점", text) or "정보 없음" in text or "근거 부족" in text
    ]

def get_analysis_score(analysis: str, checklist: List[str]) -> int:
    """분석 결과에서 점수를 추출하는 통합 함수"""
    # 먼저 총점 패턴으로 추출 시도
    score = extract_total_score_from_analysis(analysis)
    if score is not None:
        return score
    
    # 총점이 없으면 체크리스트 항목별 점수 합산
    return extract_checklist_scores(analysis, checklist)

def format_checklist_items(checklist: List[str]) -> str:
    """체크리스트 항목들을 번호와 함께 포맷팅하는 함수"""
    return "\n".join(f"{i+1}. {q}" for i, q in enumerate(checklist))

def create_analysis_prompt_template(checklist: List[str]) -> str:
    """분석용 프롬프트 템플릿을 생성하는 함수"""
    items_formatted = format_checklist_items(checklist)
    
    return """당신은 스타트업 '{startup_name}'을 다음 체크리스트 10문항에 따라 평가해야 합니다.

체크리스트:
{items}

다음 Tavily API 결과(최대 5개)를 참고하세요:
{tavily_context}

다음 Naver News API 결과(최대 5개)를 참고하세요:
{naver_context}

평가 시 유의사항:
- 정보가 부족하거나 명확하지 않을 경우 '정보 부족으로 점수 유보' 대신 관용적으로 5점 내외를 부여할 수 있습니다.
- 부정적 근거가 명확하지 않다면 초기 스타트업 상황을 고려하여 가능성에 가중치를 두고 평가하세요.

점수 부여 기준:
- 매우 우수하거나 충분히 충족 → 9~10점
- 일부 충족되었거나 불완전 → 5~8점
- 거의 정보가 없거나 충족되지 않음 → 0~4점

각 문항별 점수(0~10)와 판단 근거를 작성하세요.
판단 근거 뒤에는 관련 URL을 괄호 안에 포함하세요. URL이 없을 경우 '정보 없음'으로 표기하세요.
마지막에 총점: 숫자 (숫자만 입력, 예: 75)를 작성하세요.""" 

def rerank_with_cross_encoder(query: str, docs: list, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 5):
    """
    Cross-Encoder로 문서 리스트를 rerank하여 상위 top_k개 반환
    docs: list of dicts, each with at least 'page_content' key
    """
    from sentence_transformers import CrossEncoder
    if not docs:
        return []
    model = CrossEncoder(model_name)
    pairs = [(query, doc["page_content"]) for doc in docs]
    scores = model.predict(pairs)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in reranked[:top_k]] 

def parse_llm_json_response(llm_output: str, default: dict = None) -> dict:
    """
    LLM 응답을 JsonOutputParser로 파싱하되, 실패 시 수동 파싱/예외처리/방어적 처리를 적용하는 함수
    - 1차: JsonOutputParser로 파싱
    - 2차: 정규표현식으로 JSON 부분 추출 후 파싱
    - 3차: 실패 시 default 반환 (없으면 {'error': ...})
    """
    from langchain_core.output_parsers import JsonOutputParser
    import json, re
    parser = JsonOutputParser()
    try:
        return parser.parse(llm_output)
    except Exception as e1:
        print(f"[JsonOutputParser] 1차 파싱 실패: {e1}")
        # 코드블록/설명 등 제거, JSON 부분만 추출
        try:
            # ```json ... ``` 또는 ``` ... ``` 제거
            cleaned = llm_output.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            # 첫 번째 중괄호 블록 추출
            match = re.search(r'\{[\s\S]*\}', cleaned)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                raise ValueError("No JSON object found in LLM output")
        except Exception as e2:
            print(f"[JsonOutputParser] 2차 수동 파싱 실패: {e2}")
            if default is not None:
                return default
            return {"error": f"파싱 실패: {e1} / {e2}", "raw": llm_output} 