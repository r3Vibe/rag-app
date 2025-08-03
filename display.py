import os

import streamlit as st
from langchain_core.messages import AIMessageChunk

from document_loader import load_given_pdf
from graph import create_rag_agent

agent = create_rag_agent()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "show_upload_success" not in st.session_state:
    st.session_state.show_upload_success = False

if "upload_message" not in st.session_state:
    st.session_state.upload_message = ""

# Display previous messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question...")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file to Add Context", type="pdf")

# Handle file upload processing
if uploaded_file is not None and not st.session_state.show_upload_success:
    # Create documents folder if it doesn't exist
    documents_folder = "documents"
    if not os.path.exists(documents_folder):
        os.makedirs(documents_folder)

    # Save the uploaded file to the documents folder
    file_path = os.path.join(documents_folder, uploaded_file.name)

    # Write the uploaded file to the documents folder
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Now process the saved PDF file
    try:
        load_given_pdf(file_path)
        st.session_state.show_upload_success = True
        st.session_state.upload_message = (
            f"PDF '{uploaded_file.name}' processed and added to knowledge base!"
        )
        st.rerun()
    except Exception as e:
        st.session_state.show_upload_success = True
        st.session_state.upload_message = f"Error processing PDF: {str(e)}"
        st.rerun()

# Show success message with dismiss button
if st.session_state.show_upload_success:
    with st.sidebar:
        st.success(st.session_state.upload_message)
        if st.button("Dismiss", key="dismiss_upload_message"):
            st.session_state.show_upload_success = False
            st.session_state.upload_message = ""
            st.rerun()


if user_input:
    # Show user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare assistant message container
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        streamed_response = ""

        for chunk, _ in agent.stream(
            input={"query": user_input}, stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk):
                streamed_response += chunk.content
                response_placeholder.markdown(streamed_response)

    # Save the full response to history
    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": streamed_response,
        }
    )
