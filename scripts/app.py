import streamlit as st
import re
from chatbot import Chatbot, Message, Role
from chatbot import SourcesEvent, ChunkEvent, FinalAnswerEvent
from faiss_handler import add_pdf_to_faiss
from search_web import search_web_for_answer

# 📌 `<think>...</think>` 태그 내부 내용 추출하는 함수
def extract_thinking_process(text):
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else None

# 앱 제목
st.title("💬 AI Chatbot - Paper Writing Assistant")

# 챗봇 인스턴스 생성
chatbot = Chatbot()

# ✅ 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processing" not in st.session_state:
    st.session_state.pdf_processing = False
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# 📂 PDF 파일 업로드 기능
st.sidebar.header("📂 PDF 파일 업로드")
uploaded_file = st.sidebar.file_uploader("📄 PDF 파일을 선택하세요", type=["pdf"])

# ✅ PDF 업로드 후 처음 한 번만 처리
if uploaded_file and not st.session_state.pdf_uploaded:
    file_path = f"uploaded_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # PDF 파일을 로컬에 저장
    
    # ✅ PDF 처리 시작
    st.session_state.pdf_processing = True

    with st.spinner("🔄 PDF 처리 중..."):
        result_message = add_pdf_to_faiss(file_path)  # ✅ PDF를 FAISS DB에 추가
        st.sidebar.success(result_message)

    # ✅ PDF 처리 완료
    st.session_state.pdf_processing = False
    st.session_state.pdf_uploaded = True  # ✅ PDF가 처리되었음을 기록
    st.experimental_rerun()  # 🚨 UI 강제 리프레시

# 📝 질문 입력
user_input = st.text_area("👤 질문을 입력하세요:", height=100)

# ✅ PDF 처리 상태를 확인하여 질문 가능 여부 결정
if st.session_state.pdf_processing:
    st.warning("📄 PDF 처리 중... 질문을 입력하려면 기다려주세요.")
else:
    if st.button("💡 질문하기"):
        if user_input:
            with st.spinner("📝 답변 생성 중..."):
                chat_history = st.session_state.chat_history  # 세션 히스토리 참조
                
                # 사용자 질문을 대화 기록에 추가하고 출력
                st.session_state.chat_history.append(Message(role=Role.USER, content=user_input))
                st.chat_message("user").write(user_input)

                response_text = ""
                thinking_process = ""
                found_sources = False

                # ✅ 챗봇 응답 이벤트 처리
                for event in chatbot.ask(user_input, chat_history):
                    if isinstance(event, SourcesEvent):
                        st.session_state.chat_history.append(event)
                        found_sources = True
                    elif isinstance(event, FinalAnswerEvent):  # ✅ 최종 답변
                        response_text = event.content
                        extracted_thinking = extract_thinking_process(response_text)  # `<think>` 내부 추출
                        if extracted_thinking:
                            thinking_process = extracted_thinking  # 중간 과정 저장
                            response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
                        st.session_state.chat_history.append(Message(role=Role.ASSISTANT, content=response_text))

                # ✅ FAISS DB에서 검색되지 않았을 경우 웹 검색
                if not found_sources:
                    st.warning("📡 로컬 DB에서 관련 정보를 찾지 못했습니다. 웹에서 검색 중...")
                    web_results = search_web_for_answer(user_input)

                    if isinstance(web_results, list) and web_results:
                        st.subheader("🌍 웹 검색 결과")
                        for i, result in enumerate(web_results[:3]):
                            st.markdown(f"**{i+1}. [{result['title']}]({result['link']})**")
                            st.write(result["snippet"])
                    else:
                        st.error("❌ 웹 검색 결과를 가져오는 데 실패했습니다.")

                # ✅ Thinking Process는 접을 수 있도록 설정
                if thinking_process:
                    with st.expander("🤔 Thinking Process (중간 과정)"):
                        if re.search(r"\$.*?\$|\\\(", thinking_process):  # ✅ LaTeX 포함 여부 확인
                            st.latex(thinking_process)
                        else:
                            st.write(thinking_process)

                # ✅ 챗봇 최종 응답 출력 (최근 대화)
                if re.search(r"\$.*?\$|\\\(", response_text):  # ✅ 최종 답변에 LaTeX 포함 여부 확인
                    st.latex(response_text)
                else:
                    st.chat_message("assistant").write(response_text)
                    
                # ✅ 질문 입력창 초기화 (입력된 내용 지우기)
                st.session_state.user_input = ""


# ✅ 과거 대화 기록을 접었다 펼칠 수 있는 expander
with st.expander("📜 대화 기록 (과거 대화 접기/펼치기)"):
    for idx, msg in enumerate(st.session_state.chat_history):
        if isinstance(msg, Message):  # Message 객체만 처리
            sender = "👤 사용자" if msg.role == Role.USER else "🤖 챗봇"

            # ✅ <think> 태그 내부 내용 추출
            extracted_thinking = re.search(r"<think>(.*?)</think>", msg.content, re.DOTALL)
            thinking_text = extracted_thinking.group(1).strip() if extracted_thinking else None

            # ✅ 메시지 본문에서 <think> 제거 후 정리
            clean_content = re.sub(r"<think>.*?</think>", "", msg.content, flags=re.DOTALL).strip()

            # ✅ 메시지 기본 내용 표시
            st.markdown(f"**{sender}:** {clean_content}")

            # ✅ Thinking Process가 있으면 개별 체크박스로 접었다 펼칠 수 있도록 추가
            if thinking_text:
                show_thinking_key = f"show_thinking_{idx}"  # ✅ 고유한 키 설정

                # ✅ 체크박스 생성 (세션 상태 직접 수정 X, Streamlit이 자동 관리)
                show_thinking = st.checkbox(
                    f"🤔 {sender}의 Thinking Process 보기",
                    value=st.session_state.get(show_thinking_key, False),
                    key=show_thinking_key
                )

                # ✅ Thinking Process 내용 출력 (체크박스 선택 시)
                if show_thinking:
                    # ✅ LaTeX 포함 여부에 따라 렌더링 방식 변경
                    if re.search(r"\$.*?\$|\\\(", thinking_text):
                        st.latex(thinking_text)
                    else:
                        st.write(thinking_text)
