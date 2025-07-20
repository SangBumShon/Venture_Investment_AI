#!/usr/bin/env python3
"""
스타트업 투자 분석 시스템 - 통합 실행 스크립트
원래처럼 간단하게: python run_analysis.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# Python 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_environment():
    """환경변수 체크"""
    required_vars = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY", 
        "TAVILY_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ 다음 환경변수가 설정되지 않았습니다:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 .env 파일을 확인하거나 환경변수를 설정해주세요.")
        return False
    
    print("✅ 모든 환경변수가 설정되었습니다.")
    return True

def setup_directories():
    """필요한 디렉토리 생성"""
    directories = ["data", "investment_reports", "font"]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
    
    # PDF 파일 체크
    pdf_files = list(Path("data").glob("*.pdf"))
    if pdf_files:
        print(f"📄 PDF 파일 {len(pdf_files)}개 발견 (시장성 분석용)")
    else:
        print("⚠️  data/ 폴더에 PDF 파일이 없습니다. 시장성 분석이 제한될 수 있습니다.")

def main():
    """메인 실행 함수 - 원래 방식과 동일"""
    print("=" * 60)
    print("🚀 스타트업 투자 분석 시스템")
    print("=" * 60)
    
    # 환경 체크
    if not check_environment():
        return
    
    setup_directories()
    
    # 스타트업 이름 입력 (원래와 동일)
    startup_name = input("\n📝 분석할 스타트업 이름을 입력하세요: ").strip()
    if not startup_name:
        print("❌ 스타트업 이름이 입력되지 않았습니다.")
        return
    
    print(f"\n🔍 '{startup_name}' 투자 심사 분석 시작...")
    print("=" * 60)
    
    try:
        # 모듈 임포트 (실행 시점에 임포트)
        from app.core.graph import AnalysisGraph
        
        # 분석 그래프 생성 및 실행
        analysis_graph = AnalysisGraph()
        result = analysis_graph.analyze_startup(startup_name)
        
        # 결과 출력 (원래와 동일한 형식)
        print("\n" + "=" * 60)
        print("📊 분석 완료!")
        print("=" * 60)
        
        print(f"🏢 스타트업: {startup_name}")
        print(f"🎯 최종 판단: {result.get('최종_판단', '알 수 없음')}")
        
        # 점수 요약 (원래와 동일)
        scores = {
            "상품/서비스": result.get("상품_점수", 0),
            "기술": result.get("기술_점수", 0),
            "성장률": result.get("성장률_점수", 0),
            "시장성": result.get("시장성_점수", 0),
            "경쟁사": result.get("경쟁사_점수", 0)
        }
        
        print("\n📈 점수 요약:")
        for category, score in scores.items():
            emoji = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
            print(f"   {emoji} {category}: {score}점")
        
        avg_score = sum(scores.values()) / len(scores)
        print(f"\n📊 평균 점수: {avg_score:.1f}점")
        
        # 보고서 경로 출력 (원래와 동일)
        if result.get("pdf_path"):
            print(f"📄 보고서 생성 완료: {result['pdf_path']}")
        else:
            print("⚠️  보고서 생성에 실패했습니다.")
        
        # 보고서 미리보기 (원래와 동일)
        print("\n📋 보고서 미리보기:")
        print("-" * 40)
        report_content = result.get("보고서", "보고서 내용을 찾을 수 없습니다.")
        print(report_content[:500] + "..." if len(report_content) > 500 else report_content)
        
        print("\n" + "=" * 60)
        print("✅ 모든 분석이 완료되었습니다!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 분석 중 오류 발생: {e}")
        print("\n🔍 상세 오류 정보:")
        import traceback
        traceback.print_exc()
        
        print("\n💡 문제 해결 방법:")
        print("1. .env 파일의 API 키들을 확인하세요")
        print("2. 필요한 패키지가 모두 설치되었는지 확인하세요: pip install -r requirements.txt")
        print("3. data/ 폴더에 PDF 파일들이 있는지 확인하세요")

if __name__ == "__main__":
    main()