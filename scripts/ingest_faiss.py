from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from context_processing import _create_chunks
from extract_pdf import extract_pdfs_from_folder
from config import Config
from tqdm import tqdm
def store_documents_in_faiss():  
    """PDF 문서를 FAISS DB에 저장"""
    # all-MiniLM-L6-v2 모델 사용
    embeddings = OllamaEmbeddings(model="llama3.2")
    
    # PDF 폴더에서 문서 추출 (각 문서는 {"name": ..., "content": ...} 형태)
    documents = extract_pdfs_from_folder(str(Config.Path.DATA_DIR))
    
    all_chunks = []
    doc_idx=0
    for doc in tqdm(documents):
        # 각 PDF를 LangChain Document 객체로 생성 (메타데이터에 파일명 저장)
        doc_obj = Document(page_content=doc["content"], metadata={"source": doc["name"]})
        # 페이지 단위 혹은 원하는 기준으로 chunking
        chunks = _create_chunks(doc_obj)
        all_chunks.extend(chunks)
        doc_idx+=1
        print(f"✅ {doc_idx}번쨰 논문 Chunking이 완료되었습니다.")

    
    # FAISS 인덱스 생성 (chunk 단위 문서와 임베딩 연결)
    faiss_index = FAISS.from_documents(all_chunks, embeddings)
    
    # FAISS 인덱스를 디스크에 저장 (경로는 Config 설정 참조)
    faiss_index.save_local(str(Config.Path.FAISS_DB_DIR))
    
    print(f"✅ {len(documents)}개의 논문이 FAISS DB에 저장되었습니다.")

if __name__ == "__main__":
    store_documents_in_faiss()
