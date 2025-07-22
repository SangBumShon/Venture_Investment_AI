import os
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
# 환경변수 로드
api_key = os.getenv("PINECONE_API_KEY")
host = os.getenv("PINECONE_HOST")
index_name = os.getenv("PINECONE_INDEX_NAME")

print("API KEY:", "OK" if api_key else "MISSING")
print("HOST:", host)
print("INDEX:", index_name)

if not (api_key and host and index_name):
    raise ValueError("Pinecone API 키, 호스트, 인덱스명을 .env에 설정하세요.")

# Pinecone 연결 테스트
pinecone.init(api_key=api_key, host=host)
print("Pinecone 연결 성공!")

# 인덱스 존재 확인
print("인덱스 목록:", pinecone.list_indexes())
if index_name not in pinecone.list_indexes():
    raise ValueError(f"인덱스 '{index_name}'가 존재하지 않습니다. Pinecone 콘솔에서 생성하세요.")

# 간단한 벡터 업로드/검색 테스트
embeddings = OpenAIEmbeddings()
docs = [
    # langchain 문서 객체가 아니어도, 간단한 dict로도 테스트 가능
    # 실제로는 langchain의 Document 객체를 써야 함
    # 여기서는 예시로 간단하게 처리
    type("Doc", (), {"page_content": "Pinecone connection test", "metadata": {"source": "test"}})(),
]

# 업로드 (테스트용, 실제로는 from_documents를 사용)
try:
    PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
    print("업로드 성공!")
except Exception as e:
    print("업로드 실패:", e)

# 검색 (테스트)
try:
    vector_store = PineconeVectorStore.from_existing_index(index_name, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    results = retriever.get_relevant_documents("Pinecone connection test")
    print("검색 결과:", results)
except Exception as e:
    print("검색 실패:", e)