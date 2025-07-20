# app/services/external/api_client.py
import requests
from typing import List, Dict, Any
import os

class ExternalAPIClient:
    """외부 API 호출을 위한 클라이언트"""
    
    def __init__(self):
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.naver_id = os.getenv("NAVER_CLIENT_ID")
        self.naver_secret = os.getenv("NAVER_CLIENT_SECRET")
    
    def search_tavily(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Tavily API 검색"""
        if not self.tavily_key:
            print("Tavily API 키가 설정되지 않았습니다.")
            return []
        
        # Tavily API v1 엔드포인트 - POST 방식 사용
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.tavily_key,
            "query": query,
            "max_results": limit,
            "search_depth": "basic"
        }
        
        try:
            # POST 방식으로 요청
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 404:
                print(f"Tavily API 404 오류: 엔드포인트가 변경되었습니다.")
                return []
            elif response.status_code == 401:
                print(f"Tavily API 401 오류: API 키가 인증되지 않았습니다.")
                return []
            elif response.status_code == 405:
                print(f"Tavily API 405 오류: HTTP 메서드가 잘못되었습니다.")
                return []
            elif response.status_code != 200:
                print(f"Tavily API 오류: {response.status_code} - {response.text}")
                return []
            
            data = response.json()
            # Tavily API 응답 구조에 맞게 수정
            results = data.get("results", [])
            
            # 결과를 기존 형식으로 변환
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "published_date": result.get("published_date", ""),
                    "score": result.get("score", 0)
                })
            
            return formatted_results
            
        except requests.exceptions.Timeout:
            print("Tavily API 타임아웃")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Tavily API 요청 오류: {e}")
            return []
        except Exception as e:
            print(f"Tavily API 예상치 못한 오류: {e}")
            return []
    
    def search_naver_news(self, query: str, display: int = 5) -> List[Dict[str, Any]]:
        """Naver News API 검색"""
        if not self.naver_id or not self.naver_secret:
            print("Naver API 키가 설정되지 않았습니다.")
            return []
        
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": self.naver_id,
            "X-Naver-Client-Secret": self.naver_secret
        }
        params = {"query": query, "display": display}
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json().get("items", [])
        except requests.exceptions.Timeout:
            print("Naver API 타임아웃")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Naver API 요청 오류: {e}")
            return []
        except Exception as e:
            print(f"Naver API 예상치 못한 오류: {e}")
            return []