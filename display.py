"""
Streamlit web interface for the RAG (Retrieval-Augmented Generation) system.

This module provides a modern web-based chat interface for interacting with
the RAG agent. It supports real-time conversation, document upload functionality,
and maintains conversation history across sessions.
"""

import os

import streamlit as st
from langchain_core.messages import AIMessageChunk

from document_loader import load_given_pdf
from graph import create_rag_agent

# Initialize the RAG agent (cached for performance)
agent = create_rag_agent()

# Initialize session state variables for conversation and UI management
if "message_history" not in st.session_state:
    """Store conversation history across Streamlit reruns."""
    st.session_state["message_history"] = []

if "show_upload_success" not in st.session_state:
    """Flag to control display of upload success/error messages."""
    st.session_state.show_upload_success = False

if "upload_message" not in st.session_state:
    """Store the content of upload status messages."""
    st.session_state.upload_message = ""

# Display conversation history
# Recreate the conversation context by showing all previous messages
for message in st.session_state["message_history"]:
    """
    Render each message in the conversation with appropriate styling.
    
    Messages are displayed using Streamlit's chat_message component,
    which provides proper styling and avatars for user and assistant roles.
    """
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main user input interface
user_input = st.chat_input("Ask a question...")

# Sidebar file upload functionality
uploaded_file = st.sidebar.file_uploader("Upload a PDF file to Add Context", type="pdf")

# Handle PDF file upload and processing
if uploaded_file is not None and not st.session_state.show_upload_success:
    """
    Process uploaded PDF files and integrate them into the knowledge base.
    
    This section handles the complete pipeline for user-uploaded documents:
    1. File validation and storage preparation
    2. Saving the uploaded file to the documents directory
    3. Processing the PDF to extract text content
    4. Adding the content to the searchable vector store
    5. Providing user feedback on the operation status
    
    The condition ensures files are only processed once and prevents
    repeated processing during UI updates.
    """

    # Ensure the documents directory exists
    documents_folder = "documents"
    if not os.path.exists(documents_folder):
        os.makedirs(documents_folder)

    # Save the uploaded file to the documents directory
    file_path = os.path.join(documents_folder, uploaded_file.name)

    # Write the file content to disk
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the PDF and add it to the vector store
    try:
        load_given_pdf(file_path)  # Extract text and add to searchable index

        # Set success message and flag
        st.session_state.show_upload_success = True
        st.session_state.upload_message = (
            f"PDF '{uploaded_file.name}' processed and added to knowledge base!"
        )
        st.rerun()  # Refresh UI to show success message

    except Exception as e:
        # Handle processing errors with user-friendly messages
        st.session_state.show_upload_success = True
        st.session_state.upload_message = f"Error processing PDF: {str(e)}"
        st.rerun()  # Refresh UI to show error message

# Display upload status messages with dismiss functionality
if st.session_state.show_upload_success:
    """
    Show upload status messages with user-controlled dismissal.
    
    This provides feedback about file upload operations and allows users
    to dismiss notifications when they're done reading them. The dismiss
    functionality helps keep the UI clean and uncluttered.
    """
    with st.sidebar:
        st.success(st.session_state.upload_message)

        # Provide dismiss button with unique key for proper state handling
        if st.button("Dismiss", key="dismiss_upload_message"):
            # Clear the message state and refresh UI
            st.session_state.show_upload_success = False
            st.session_state.upload_message = ""
            st.rerun()


# Handle user queries and generate responses
if user_input:
    """
    Process user input and generate AI responses using the RAG system.
    
    This section handles the complete conversation flow:
    1. Store and display the user's message
    2. Stream the AI response in real-time
    3. Update the conversation history
    4. Maintain proper message formatting and state
    
    The streaming approach provides immediate feedback and shows the AI's
    response as it's being generated, creating a more engaging user experience.
    """

    # Add user message to conversation history
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    # Display the user's message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and stream the AI response
    with st.chat_message("assistant"):
        """
        Create a streaming response display that updates in real-time.
        
        This approach:
        - Shows immediate feedback to the user
        - Updates the display as new content arrives
        - Provides a smooth conversational experience
        - Handles message chunks properly for streaming
        """
        response_placeholder = st.empty()  # Placeholder for dynamic content updates
        streamed_response = ""  # Accumulator for the complete response

        # Stream the response from the RAG agent
        for chunk, _ in agent.stream(
            input={"query": user_input}, stream_mode="messages"
        ):
            # Process AI message chunks and accumulate content
            if isinstance(chunk, AIMessageChunk):
                streamed_response += chunk.content
                # Update the display with accumulated content
                response_placeholder.markdown(streamed_response)

    # Save the complete response to conversation history
    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": streamed_response,
        }
    )
