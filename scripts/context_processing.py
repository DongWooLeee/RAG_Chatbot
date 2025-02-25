from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import Config

# 📌 청크별 컨텍스트 생성 프롬프트
CONTEXT_PROMPT = ChatPromptTemplate.from_template(
    """
    You're an expert in document analysis. Provide a concise context (2-3 sentences) for this chunk.

    <document>
    {document}
    </document>

    <chunk>
    {chunk}
    </chunk>
    
    Provide a concise context (2-3 sentences) for this chunk, considering the following guidelines:
    1. Identify the main topic or concept discussed in the chunk.
    2. Mention any relevant information or comparisons from the broader document context.
    3. If applicable, note how this information relates to the overall theme or purpose of the document.
    4. Include any key figures, dates, or percentages that provide important context.
    5. Do not use phrases like "This chunk discusses..." or "The chunk is about...". Instead, directly state that information.
    
    Please give a short succint context to situate this chunk within the overall document for the purpose of summarization.

    Context:
    """.strip()
)
# RecursiveCharacterTextSplitter: 재귀적 문자 기반 텍스트 분할기. 문서를 일정한 크기(chunk_size)로 나누되, 청크 간 chunk_overlap 만큼 중첩을 유지.
# RecursiveCharacterTextSplitter는 문장 구조를 유지하면서 효율적으로 청킹.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=Config.Preprocessing.CHUNK_SIZE,
    chunk_overlap=Config.Preprocessing.CHUNK_OVERLAP
)

def create_llm() -> ChatOllama:
    """Ollama LLM 생성"""
    return ChatOllama(
        model=Config.Preprocessing.LLM, 
        temperature=Config.Model.TEMPERATURE, 
        seed=Config.SEED, 
        keep_alive=-1
    )

def _generate_context(llm: ChatOllama, document: str, chunk: str) -> str:
    """LLM을 사용하여 청크에 대한 컨텍스트 생성"""
    messages = CONTEXT_PROMPT.format_messages(document=document, chunk=chunk)
    response = llm.invoke(messages)
    return response.content

# # _generate_context()를 활용해 각 청크에 문맥 정보를 추가.
def _create_chunks(document: Document) -> list:
    """문서를 청크로 분할 후 컨텍스트 추가"""
    chunks = text_splitter.split_documents([document])

    if not Config.Preprocessing.CONTEXTUALIZE_CHUNKS:
        return chunks

    llm = create_llm()
    contextual_chunks = []

    for chunk in chunks:
        context = _generate_context(llm, document.page_content, chunk.page_content)
        chunk_with_context = f"{context}\n\n{chunk.page_content}"
        contextual_chunks.append(Document(page_content=chunk_with_context, metadata=chunk.metadata))

    return contextual_chunks
