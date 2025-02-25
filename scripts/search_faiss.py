from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
#from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config import Config
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank


FAISS_PATH = str(Config.Path.FAISS_DB_DIR)

def search_faiss(query, top_k=Config.Preprocessing.N_SEMANTIC_CHUNKS):
    """BM25 + Semantic 검색 (FAISS 기반)"""
    embeddings = OllamaEmbeddings(model="llama3.2")
    # 디스크에서 FAISS 인덱스 로드
    faiss_index = FAISS.load_local(FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
    
    # FAISS에 저장된 모든 Document 객체 추출
    all_docs = list(faiss_index.docstore._dict.values())  # 내부 dict에서 문서 가져오기

    
    # BM25 기반 키워드 검색기 (문서 단위)
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = top_k
    
    # FAISS를 활용한 Semantic 검색기
    semantic_retriever = faiss_index.as_retriever(search_kwargs={"k": top_k})
    
    # 두 검색 결과를 Ensemble 방식으로 결합 (BM25: 60%, Semantic: 40%)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.6, 0.4]
)
    
    search_results = ensemble_retriever.invoke(query)
    
    ## FlashrankRerank 사용하여 결과 재정렬
    reranker = FlashrankRerank(model=Config.Preprocessing.RERANKER)  # ColBERT 기반 Ranker 사용
    
    reranked_results = reranker.compress_documents(search_results, query=query)
    
    reranked_results = reranked_results[:top_k]

    
    return reranked_results
