import os
from pathlib import Path

class Config:
    
    SEED = 42
    ALLOWED_FILE_EXTENSIONS = {".pdf", ".md", ".txt", ".docx"}  # 지원하는 파일 확장자
    
    class Model:
        NAME = "deepseek-r1:8b"
        TEMPERATURE = 0.1  # LLM 응답 다양성 조절
        
    class Preprocessing:
        CHUNK_SIZE = 2048
        CHUNK_OVERLAP = 128
        EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
        RERANKER = "ms-marco-MiniLM-L-12-v2"
        LLM = "llama3.2"
        CONTEXTUALIZE_CHUNKS = True
        N_SEMANTIC_CHUNKS = 5  # 의미적 검색 시 반환할 최대 청크 수
        N_BM25_RESULTS = 5  # BM25 검색 시 반환할 결과 개수
        
    class Chatbot:
        N_CONTEXT_RESULTS = 10  # 챗봇이 제공하는 문맥 수
        
    class Path:
        APP_HOME = '/workspace/dongwoo/chatbot_project'
        DATA_DIR = '/workspace/dongwoo/chatbot_project/data/pdf'
        FAISS_DB_DIR = '/workspace/dongwoo/chatbot_project/faiss_db'
