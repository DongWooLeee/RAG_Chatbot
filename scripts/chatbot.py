from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from typing import List, Union, Iterable
from enum import Enum
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from search_faiss import search_faiss  # FAISS 기반 검색 함수 사용
from config import Config
#langraph
# langgraph
#langgraph.graph: LangGraph의 상태 머신을 정의하는 데 사용.
#langchain_core.messages: LLM과 주고받는 메시지를 정의.
#langchain_core.prompts: LLM에게 전달할 프롬프트를 구성.
#dataclasses: @dataclass를 사용해 간단한 데이터 구조를 정의.
#search_faiss: FAISS 기반 유사도 검색 함수 (사용자 정의 모듈).
#config: 설정 값이 저장된 모듈.



# 시스템 프롬프트 (LLM이 따를 기본 규칙)
SYSTEM_PROMPT = """
    You're having a conversation with an user about excerpts of their files.
    Try to be helpful and answer their questions.
    
    If you do not know the answer, say that you do not know and try to clarify the question.
""".strip()

# 문서 기반 답변 생성 템플릿
PROMPT = """
Here's the information you have about the excerpts of the files:

<context>
{context}
</context>

One file can have multiple excerpts.

Please, respond to the query below:

<question>
{question}
</question>

Answer:
"""
# LLM이 참고할 문서 내용을 <context> 안에 넣음.
# <question> 안에는 사용자의 질문을 삽입.
#FAISS에서 검색한 문서를 바탕으로 답변을 생성.

# 전체 대화 흐름을 위한 프롬프트 템플릿
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", PROMPT),
    ]
)

# 역할 Enum
class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"

# 대화 메시지 저장용 데이터 클래스
@dataclass
class Message:
    role: Role
    content: str

# 검색된 문서 목록 저장 (문서 이름 리스트)
@dataclass
class SourcesEvent:
    sources: List[str]

# 스트리밍된 답변 청크 저장 (부분 답변)
@dataclass
class ChunkEvent:
    content: str

# 최종 생성된 LLM 답변 저장
@dataclass
class FinalAnswerEvent:
    content: str

# LangGraph 상태 정의
class State(dict):
    question: str
    chat_history: List[BaseMessage]
    context: List[str]
    answer: str

# FAISS 기반 챗봇 클래스
class Chatbot:
    def __init__(self):
        self.llm = ChatOllama(
            model=Config.Model.NAME,
            temperature=Config.Model.TEMPERATURE,
            seed=Config.SEED,
            keep_alive=-1,
            verbose=False
        )
        self.workflow = self._create_workflow()
        self.topk = Config.Preprocessing.N_SEMANTIC_CHUNKS

    def _format_docs(self, docs: List[str]) -> str:
        """문서들을 줄바꿈으로 연결하여 하나의 문자열로 포맷"""
        return "\n\n".join(docs)

    def _retrieve(self, state: State):
        """FAISS DB에서 문서 검색 후 본문 데이터 포함"""
        results = search_faiss(state["question"], top_k=self.topk)
        context = [doc.page_content for doc in results]  # 검색된 각 문서의 본문
        return {"context": context}

    def _generate(self, state: State):
        """LLM을 사용해 답변 생성"""
        messages = PROMPT_TEMPLATE.invoke(
            {
                "question": state["question"],
                "context": self._format_docs(state["context"]),
                "chat_history": state["chat_history"],
            }
        )
        answer = self.llm.invoke(messages)
        return {"answer": answer}

    def _create_workflow(self) -> CompiledStateGraph:
        """LangGraph 기반 챗봇 워크플로우 생성"""
        graph_builder = StateGraph(State).add_sequence([self._retrieve, self._generate])
        graph_builder.add_edge(START, "_retrieve")
        return graph_builder.compile()

    def _ask_model(self, prompt: str, chat_history: List[Message]) -> Iterable[Union[SourcesEvent, ChunkEvent, FinalAnswerEvent]]:
        # 대화 기록에서 Message 객체만 필터링하여 사용
        history = [
            AIMessage(m.content) if m.role == Role.ASSISTANT else HumanMessage(m.content)
            for m in chat_history if isinstance(m, Message)
        ]
        payload = {"question": prompt, "chat_history": history}

        for event_type, event_data in self.workflow.stream(
            payload,
            config={"configurable": {"thread_id": 42}},
            stream_mode=["updates", "messages"],
        ):
            if event_type == "messages":
                chunk, _ = event_data
                yield ChunkEvent(chunk.content)
            if event_type == "updates":
                if "_retrieve" in event_data:
                    documents = event_data["_retrieve"]["context"]
                    yield SourcesEvent(sources=documents)
                if "_generate" in event_data:
                    answer = event_data["_generate"]["answer"]
                    yield FinalAnswerEvent(content=answer.content)

    def ask(
        self, prompt: str, chat_history: List[Message]
    ) -> Iterable[Union[SourcesEvent, ChunkEvent, FinalAnswerEvent]]:
        """챗봇에 질문을 입력하고 답변을 받아오는 함수"""
        for event in self._ask_model(prompt, chat_history):
            yield event
            if isinstance(event, FinalAnswerEvent):
                chat_history.append(Message(role=Role.USER, content=prompt))
                chat_history.append(Message(role=Role.ASSISTANT, content=event.content))
