import time

import streamlit as st
from langchain_core.messages import AIMessageChunk

from graph import create_rag_agent

st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.title("RAG Chat Assistant 🤖")


pdf = st.sidebar.file_uploader(
    label="Upload a PDF file", type=["pdf"], key="pdf_uploader"
)

if pdf is not None:
    file_path = f"documents/{pdf.name}"
    with st.sidebar:
        with st.spinner("Loading PDF ..."):
            with open(file_path, "wb") as f:
                f.write(pdf.getbuffer())
        s1 = st.success("PDF loaded successfully!")

        with st.spinner("Processing PDF ..."):
            from document_loader import load_given_pdf

            load_given_pdf(file_path)
        s2 = st.success("PDF processed successfully!")
        s3 = st.success("You can now ask questions about the content of the PDF.")

        time.sleep(1)
        s1.empty()
        s2.empty()
        s3.empty()


user_input = st.chat_input("Ask a question...")
agent = create_rag_agent()

# Session state setup
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Display chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input:
    import uuid

    thread_id = str(uuid.uuid4().hex)
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        streamed_response = ""

        for chunk, _ in agent.stream(
            input={"query": user_input},
            stream_mode="messages",
            config={"configurable": {"thread_id": thread_id}},
        ):
            if isinstance(chunk, AIMessageChunk):
                streamed_response += chunk.content
                response_placeholder.markdown(streamed_response)

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": streamed_response,
        }
    )
