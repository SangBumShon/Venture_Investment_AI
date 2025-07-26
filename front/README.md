# AI 투자 평가 대시보드 (Frontend)

AI 스타트업 투자 평가 시스템의 Vue.js 프론트엔드입니다.

## 🚀 실행 방법

### 1. 의존성 설치
```bash
cd front
npm install
```

### 2. 개발 서버 실행
```bash
npm run dev
```

### 3. 브라우저에서 확인
- http://localhost:3000 으로 접속

## 📦 주요 패키지

- **Vue 3**: 프론트엔드 프레임워크
- **Vite**: 빌드 도구
- **Tailwind CSS**: CSS 프레임워크
- **Font Awesome**: 아이콘 라이브러리

## 🔧 개발 명령어

```bash
# 개발 서버 실행
npm run dev

# 프로덕션 빌드
npm run build

# 빌드 결과 미리보기
npm run preview
```

## 🌐 API 연동

- 백엔드 API 서버 (FastAPI)와 연동
- 프록시 설정: `/api` → `http://localhost:8000`
- 실시간 분석 진행상황 추적
- PDF 보고서 다운로드

## 📁 프로젝트 구조

```
front/
├── src/
│   ├── components/
│   │   └── AI-투자-평가-대시보드.vue  # 메인 대시보드 컴포넌트
│   ├── App.vue                        # 루트 컴포넌트
│   ├── main.js                        # 앱 진입점
│   └── style.css                      # 전역 스타일
├── index.html                         # 메인 HTML
├── package.json                       # 의존성 정의
├── vite.config.js                     # Vite 설정
├── tailwind.config.js                 # Tailwind 설정
└── postcss.config.js                  # PostCSS 설정
``` 