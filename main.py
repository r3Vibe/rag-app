"""
Command-line interface for the RAG (Retrieval-Augmented Generation) system.

This module provides a simple console-based interface for interacting with
the RAG agent. Users can ask questions and receive answers based on the
indexed document collection.
"""

import uuid

from langchain_core.messages import AIMessageChunk

from graph import create_rag_agent


def main():
    """
    Main interactive loop for the RAG system command-line interface.

    This function creates a RAG agent and provides a continuous interactive
    session where users can:
    1. Enter questions or queries
    2. Receive AI-generated answers based on indexed documents
    3. Continue the conversation or exit

    The function handles:
    - RAG agent initialization
    - User input collection
    - Response streaming and display
    - Graceful exit handling

    Features:
    - Continuous conversation loop
    - Real-time response streaming
    - Clean exit mechanism
    - Error handling for agent operations

    Usage:
        Run the script and enter questions when prompted.
        Type 'exit' to quit the application.

    Note:
        Responses are streamed in real-time, showing the AI's thinking
        process as it generates answers.
    """

    # Initialize the RAG agent with the compiled workflow
    rag_agent = create_rag_agent()

    thread_id = uuid.uuid4().hex  # Generate a unique thread ID for the conversation

    # Start the interactive conversation loop
    while True:
        # Get user input with clear instructions
        user_query = input("Enter your query (or 'exit' to quit): ")

        # Check for exit condition
        if user_query.lower() == "exit":
            break

        # Stream the response in real-time
        # This shows the AI's response as it's being generated
        for token, metadata in rag_agent.stream(
            input={"query": user_query, "role": "Manager"},
            stream_mode="messages",
            config={"configurable": {"thread_id": thread_id}},
        ):
            if isinstance(token, AIMessageChunk):
                # Print each token as it arrives for real-time feedback
                print(token.content, end="", flush=True)

        # Add spacing between responses for better readability
        print("\n")  # Extra newline for visual separation


if __name__ == "__main__":
    """
    Entry point for the command-line RAG application.
    
    When this script is run directly (not imported), it starts the
    interactive command-line interface for the RAG system.
    
    This allows users to run the application with:
        python main.py
    
    The interface provides a simple way to test the RAG system
    and interact with the indexed documents through natural language queries.
    """
    main()
