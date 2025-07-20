# app/services/external/rag_service.py (Host 정보 포함 버전)
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class RAGService:
    """RAG 검색 서비스 (Pinecone + Ensemble Retriever + CrossEncoder Reranker)"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.reranker = self._init_crossencoder()
        self.pc = None
        self.index = None
        self.doc_texts = []
        self.all_docs = []
        self._init_pinecone()
    
    def _init_crossencoder(self):
        """CrossEncoder Reranker 초기화"""
        try:
            # 한국어도 잘 처리하는 다국어 모델 사용
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            reranker = CrossEncoder(model_name)
            print(f"[INFO] CrossEncoder Reranker 로드 완료: {model_name}")
            return reranker
        except Exception as e:
            print(f"[ERROR] CrossEncoder 초기화 실패: {e}")
            return None
    
    def _init_pinecone(self):
        """Pinecone 초기화 (Host 정보 포함)"""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("[WARNING] PINECONE_API_KEY가 설정되지 않았습니다.")
            return
        
        try:
            # Pinecone 클라이언트 초기화
            self.pc = Pinecone(api_key=api_key)
            
            index_name = os.getenv("PINECONE_INDEX_NAME", "startup-analysis")
            
            # 기존 인덱스 목록에서 확인
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if index_name not in existing_indexes:
                print(f"[INFO] Pinecone 인덱스 '{index_name}' 생성 중...")
                
                # ServerlessSpec으로 생성
                self.pc.create_index(
                    name=index_name,
                    dimension=1536,  # text-embedding-3-small dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
                    )
                )
                print(f"[INFO] Pinecone 인덱스 '{index_name}' 생성 완료")
            
            # Host URL로 직접 연결 시도 (설정된 경우)
            pinecone_host = os.getenv("PINECONE_HOST")
            if pinecone_host:
                # Host가 설정된 경우 직접 연결
                self.index = self.pc.Index(index_name, host=pinecone_host)
                print(f"[INFO] Pinecone 인덱스 '{index_name}' Host 연결: {pinecone_host}")
            else:
                # Host가 없는 경우 일반 연결
                self.index = self.pc.Index(index_name)
                print(f"[INFO] Pinecone 인덱스 '{index_name}' 일반 연결 완료")
            
            # 연결 테스트
            try:
                stats = self.index.describe_index_stats()
                print(f"[INFO] 인덱스 상태: {stats.total_vector_count}개 벡터 저장됨")
            except Exception as e:
                print(f"[WARNING] 인덱스 상태 확인 실패: {e}")
                # 상태 확인 실패해도 계속 진행
            
        except Exception as e:
            print(f"[ERROR] Pinecone 초기화 실패: {e}")
            self.pc = None
            self.index = None
            
            # 상세 오류 정보 출력
            print("\n💡 Pinecone 설정 확인 사항:")
            print("1. PINECONE_API_KEY가 올바른지 확인")
            print("2. PINECONE_INDEX_NAME이 올바른지 확인")
            print("3. PINECONE_ENVIRONMENT가 올바른지 확인")
            print("4. 필요시 PINECONE_HOST를 설정")
    
    def _generate_doc_id(self, doc_content: str, metadata: dict) -> str:
        """문서 ID 생성"""
        content_hash = hashlib.md5(doc_content.encode()).hexdigest()[:8]
        source = metadata.get('source', 'unknown')
        page = metadata.get('page', 0)
        return f"{Path(source).stem}_page{page}_{content_hash}"
    
    def load_and_index_documents(self, pdf_dir: Path) -> List[str]:
        """PDF 문서 로드 및 인덱싱"""
        if not pdf_dir.exists():
            print(f"[WARNING] PDF 디렉토리가 존재하지 않습니다: {pdf_dir}")
            return []
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"[WARNING] PDF 파일이 없습니다: {pdf_dir}")
            return []
        
        all_docs = []
        doc_texts = []
        
        print(f"[DEBUG] PDF 파일 {len(pdf_files)}개 처리 시작...")
        print(f"[DEBUG] PDF 파일 목록: {[f.name for f in pdf_files]}")
        
        for pdf_file in pdf_files:
            try:
                print(f"[DEBUG] '{pdf_file.name}' 처리 중...")
                loader = PyMuPDFLoader(str(pdf_file))
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=100
                )
                docs = loader.load_and_split(text_splitter)
                
                print(f"[DEBUG] '{pdf_file.name}' → {len(docs)} 청크 생성")
                if docs:
                    print(f"[DEBUG] 첫 번째 청크 샘플: {docs[0].page_content[:100]}...")
                
                # 메타데이터 보강
                for doc in docs:
                    doc.metadata['source'] = str(pdf_file)
                    doc.metadata['filename'] = pdf_file.name
                
                all_docs.extend(docs)
                doc_texts.extend([doc.page_content for doc in docs])
                
            except Exception as e:
                print(f"[ERROR] PDF 로드 실패 '{pdf_file.name}': {e}")
                import traceback
                traceback.print_exc()
        
        # 인스턴스 변수에 저장
        self.all_docs = all_docs
        self.doc_texts = doc_texts
        
        print(f"[DEBUG] 총 {len(all_docs)}개 문서 청크 처리 완료")
        print(f"[DEBUG] Pinecone 인덱스 상태: {self.index is not None}")
        print(f"[DEBUG] 임베딩 모델 상태: {self.embeddings is not None}")
        
        # Pinecone에 업로드
        if self.index and all_docs:
            print(f"[DEBUG] Pinecone 업로드 시작...")
            self._upload_to_pinecone(all_docs)
        else:
            print(f"[DEBUG] Pinecone 업로드 건너뜀 - index: {self.index is not None}, docs: {len(all_docs)}")
        
        return doc_texts
    
    def _upload_to_pinecone(self, docs: List[Document], batch_size: int = 100):
        """Pinecone에 문서 업로드"""
        print(f"[INFO] Pinecone에 {len(docs)}개 문서 업로드 시작...")
        
        try:
            vectors_to_upload = []
            successful_embeddings = 0
            failed_embeddings = 0
            
            print(f"[DEBUG] 임베딩 모델 정보: {type(self.embeddings)}")
            
            for i, doc in enumerate(docs):
                print(f"[DEBUG] 문서 {i+1}/{len(docs)} 처리 중...")
                print(f"[DEBUG] 문서 내용 길이: {len(doc.page_content)} 문자")
                print(f"[DEBUG] 문서 내용 샘플: {doc.page_content[:100]}...")
                
                doc_id = self._generate_doc_id(doc.page_content, doc.metadata)
                
                # 임베딩 생성
                try:
                    print(f"[DEBUG] 임베딩 생성 시도...")
                    embedding = self.embeddings.embed_query(doc.page_content)
                    print(f"[DEBUG] 임베딩 생성 성공 - 차원: {len(embedding)}")
                    print(f"[DEBUG] 임베딩 샘플: {embedding[:5]}...")
                    
                    vector_data = {
                        "id": doc_id,
                        "values": embedding,
                        "metadata": {
                            "text": doc.page_content,
                            "source": doc.metadata.get('source', ''),
                            "page": doc.metadata.get('page', 0),
                            "filename": doc.metadata.get('filename', '')
                        }
                    }
                    vectors_to_upload.append(vector_data)
                    successful_embeddings += 1
                    
                except Exception as e:
                    print(f"[ERROR] 임베딩 생성 실패: {e}")
                    import traceback
                    traceback.print_exc()
                    failed_embeddings += 1
                    continue
                
                # 배치 업로드
                if len(vectors_to_upload) >= batch_size:
                    print(f"[DEBUG] 배치 업로드 실행 ({len(vectors_to_upload)}개)...")
                    self._batch_upload(vectors_to_upload)
                    vectors_to_upload = []
            
            # 남은 벡터 업로드
            if vectors_to_upload:
                print(f"[DEBUG] 마지막 배치 업로드 실행 ({len(vectors_to_upload)}개)...")
                self._batch_upload(vectors_to_upload)
            
            print(f"[INFO] Pinecone 업로드 완료 - 성공: {successful_embeddings}, 실패: {failed_embeddings}")
            
        except Exception as e:
            print(f"[ERROR] Pinecone 업로드 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def _batch_upload(self, vectors: List[Dict]):
        """배치 단위로 Pinecone에 업로드"""
        try:
            self.index.upsert(vectors=vectors)
            print(f"[DEBUG] {len(vectors)}개 벡터 업로드 완료")
        except Exception as e:
            print(f"[ERROR] 배치 업로드 실패: {e}")
    
    def _search_pinecone(self, query: str, k: int = 10) -> List[Document]:
        """Pinecone에서 유사도 검색"""
        if not self.index:
            return []
        
        try:
            # 쿼리 임베딩
            query_embedding = self.embeddings.embed_query(query)
            
            # Pinecone 검색
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )
            
            # Document 객체로 변환
            docs = []
            for match in results.matches:
                if match.metadata and 'text' in match.metadata:
                    doc = Document(
                        page_content=match.metadata['text'],
                        metadata={
                            'source': match.metadata.get('source', ''),
                            'page': match.metadata.get('page', 0),
                            'filename': match.metadata.get('filename', ''),
                            'score': match.score
                        }
                    )
                    docs.append(doc)
            
            print(f"[DEBUG] Pinecone에서 {len(docs)}개 문서 검색됨")
            return docs
            
        except Exception as e:
            print(f"[ERROR] Pinecone 검색 실패: {e}")
            return []
    
    def _crossencoder_rerank(self, query: str, documents: List[str], top_k: int = 4) -> List[str]:
        """CrossEncoder로 문서 재순위화"""
        if not self.reranker or not documents:
            return documents[:top_k]
        
        try:
            # 쿼리-문서 쌍 생성
            query_doc_pairs = [[query, doc] for doc in documents]
            
            # CrossEncoder로 점수 계산
            scores = self.reranker.predict(query_doc_pairs)
            
            # 점수와 문서를 함께 정렬
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 k개 문서 반환
            reranked_docs = [doc for doc, score in scored_docs[:top_k]]
            
            print(f"[DEBUG] CrossEncoder 재순위화: {len(documents)}개 → {len(reranked_docs)}개")
            return reranked_docs
            
        except Exception as e:
            print(f"[ERROR] CrossEncoder 재순위화 실패: {e}")
            return documents[:top_k]
    
    def create_ensemble_retriever(self, query: str, k: int = 4) -> List[str]:
        """Ensemble Retriever (Dense + Sparse + CrossEncoder Reranker)"""
        sparse_results = []
        dense_results = []
        
        # 1. BM25 (Sparse) Retriever
        if self.doc_texts:
            try:
                bm25_retriever = BM25Retriever.from_texts(self.doc_texts)
                bm25_retriever.k = k * 3  # 더 많이 가져와서 다양성 확보
                sparse_results = bm25_retriever.get_relevant_documents(query)
                print(f"[DEBUG] BM25에서 {len(sparse_results)}개 문서 검색됨")
            except Exception as e:
                print(f"[ERROR] BM25 검색 실패: {e}")
        
        # 2. Dense Retriever (Pinecone)
        dense_results = self._search_pinecone(query, k * 3)
        
        # 3. 결과 합치기 및 중복 제거
        all_results = []
        seen_contents = set()
        
        # Sparse 결과 추가
        for doc in sparse_results:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_results.append(doc)
        
        # Dense 결과 추가
        for doc in dense_results:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_results.append(doc)
        
        print(f"[DEBUG] Ensemble 결과: 총 {len(all_results)}개 문서")
        
        # 4. CrossEncoder Reranker 적용
        if all_results:
            documents = [doc.page_content for doc in all_results]
            reranked_docs = self._crossencoder_rerank(query, documents, top_k=k)
            return reranked_docs
        
        return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Pinecone 인덱스 통계 조회"""
        if not self.index:
            return {"error": "Pinecone 인덱스가 연결되지 않았습니다."}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
        except Exception as e:
            return {"error": f"통계 조회 실패: {e}"}
    
    def clear_index(self):
        """Pinecone 인덱스 초기화 (개발용)"""
        if not self.index:
            print("[WARNING] Pinecone 인덱스가 연결되지 않았습니다.")
            return
        
        try:
            self.index.delete(delete_all=True)
            print("[INFO] Pinecone 인덱스 초기화 완료")
        except Exception as e:
            print(f"[ERROR] 인덱스 초기화 실패: {e}")

    def check_pinecone_config(self):
        """Pinecone 설정 상태 체크"""
        required_vars = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
        optional_vars = ["PINECONE_ENVIRONMENT", "PINECONE_HOST"]
        
        print("\n🔍 Pinecone 설정 확인:")
        
        # 필수 변수 체크
        missing_required = []
        for var in required_vars:
            value = os.getenv(var)
            if value:
                print(f"✅ {var}: 설정됨")
            else:
                print(f"❌ {var}: 누락")
                missing_required.append(var)
        
        # 선택적 변수 체크
        for var in optional_vars:
            value = os.getenv(var)
            if value:
                print(f"✅ {var}: {value}")
            else:
                print(f"⚠️  {var}: 설정되지 않음 (기본값 사용)")
        
        if missing_required:
            print(f"\n❌ 필수 환경변수 누락: {', '.join(missing_required)}")
            return False
        
        return True