import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from context_processing import _create_chunks
from extract_pdf import extract_pdf  # ✅ 폴더 대신 단일 파일 처리 함수로 변경
from config import Config

FAISS_DB_PATH = str(Config.Path.FAISS_DB_DIR)

def load_faiss_index():
    """기존 FAISS 인덱스를 로드하거나 없으면 None 반환"""
    embeddings = OllamaEmbeddings(model="llama3.2")
    if os.path.exists(FAISS_DB_PATH):
        return FAISS.load_local(FAISS_DB_PATH, embeddings,allow_dangerous_deserialization=True)
    return None

def save_faiss_index(vectorstore):
    """업데이트된 FAISS 인덱스를 디스크에 저장"""
    vectorstore.save_local(FAISS_DB_PATH)

def add_pdf_to_faiss(pdf_file_path):
    """✅ 하나의 PDF 파일을 FAISS DB에 추가 (폴더 X, 개별 파일 O)"""
    documents = extract_pdf(pdf_file_path)  # ✅ 단일 PDF 처리
    if not documents:
        return "❌ PDF에서 텍스트를 추출하지 못했습니다."

    embeddings = OllamaEmbeddings(model="llama3.2")

    all_chunks = []
    for doc in documents:
        doc_obj = Document(page_content=doc["content"], metadata={"source": doc["name"]})
        chunks = _create_chunks(doc_obj)
        all_chunks.extend(chunks)

    vectorstore = load_faiss_index()
    if vectorstore:
        vectorstore.add_documents(all_chunks)
    else:
        vectorstore = FAISS.from_documents(all_chunks, embeddings)

    save_faiss_index(vectorstore)

    return f"✅ {pdf_file_path} 파일이 FAISS DB에 추가되었습니다!"
