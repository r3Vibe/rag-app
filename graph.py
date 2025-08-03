"""
LangGraph implementation for a RAG (Retrieval-Augmented Generation) system.

This module creates a processing graph that combines document retrieval with
language model generation to provide contextually relevant answers. The system
retrieves relevant documents based on user queries and uses them to generate
informed responses.
"""

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from llm import get_llm
from query import query_vector_store
from states import InputState, OutputState, OverallState


def create_rag_agent():
    """
    Create and configure a RAG (Retrieval-Augmented Generation) agent using LangGraph.

    This function builds a processing pipeline that:
    1. Retrieves relevant documents based on user queries
    2. Generates contextually informed responses using an LLM
    3. Formats and returns the final answer with citations

    The agent implements a two-step workflow:
    - Document retrieval and LLM generation (call_model)
    - Response formatting and finalization (format_output)

    Returns:
        CompiledGraph: A compiled LangGraph workflow ready for execution.
                      Supports both streaming and non-streaming modes.
                      Can be invoked with queries and returns structured responses.

    Architecture:
        Input -> Document Retrieval -> LLM Generation -> Output Formatting -> Response

    Features:
        - Semantic document search using FAISS vector store
        - Context-aware response generation
        - Automatic citation inclusion
        - Fallback handling for insufficient context
        - Configurable response streaming
    """

    # Initialize the language model for text generation
    llm = get_llm()

    def call_model(state: OverallState):
        """
        Core processing function that handles document retrieval and LLM generation.

        This function represents the main processing step in the RAG pipeline:
        1. Extracts the user query from the current state
        2. Performs semantic search to find relevant documents
        3. Constructs a system prompt with retrieved context
        4. Generates a response using the LLM
        5. Updates the state with the generated response

        Args:
            state (OverallState): Current workflow state containing:
                                - query: User's question
                                - messages: Conversation history
                                - Other workflow data

        Returns:
            OverallState: Updated state with the LLM's response added to messages.

        Process:
            - Creates a HumanMessage from the user query
            - Retrieves relevant document context using semantic search
            - Builds a comprehensive system prompt with instructions and context
            - Invokes the LLM with system prompt and conversation history
            - Appends the response to the message history
        """
        # Extract existing messages from state (for conversation continuity)
        local_messages = state.get("messages", [])

        # Create a human message from the current query
        human_message = HumanMessage(content=state["query"])
        local_messages.append(human_message)

        # Retrieve relevant documents using semantic similarity search
        context = query_vector_store(state["query"])

        # Construct a comprehensive system prompt with context and instructions
        system_message = SystemMessage(
            content=f"""
            You are a helpful assistant. You will answer the user's query based on the following context:
            {context}
            Please provide a concise and accurate response.
            In the end of the response, include citation for the documents used to answer the query.
            If you don't have enough information, say "I don't know" or "I need more information to answer this query."
            You are to only use the context provided and not any external knowledge.
            If the query is not related to the context do not have to provide any citation.
            In the citation only use the pdf file name and the page number.
            """
        )

        # Generate response using the LLM with system prompt and conversation history
        response = llm.invoke([system_message] + local_messages)

        # Add the LLM's response to the state's message history
        state["messages"].append(response)

        return state

    def format_output(state: OverallState):
        """
        Format the final output by extracting the answer from the last message.

        This function performs the final step in the RAG pipeline by extracting
        the generated response content and making it available in the output state.

        Args:
            state (OverallState): State containing the complete conversation history
                                with the LLM's response as the last message.

        Returns:
            OverallState: Updated state with the formatted answer ready for output.

        Note:
            This step ensures that the final answer is properly extracted and
            available in the expected format for downstream consumers.
        """
        # Extract the content from the most recent message (LLM's response)
        state["answer"] = state["messages"][-1].content
        return state

    # Define the processing graph structure using LangGraph
    graph = StateGraph(OverallState, input=InputState, output=OutputState)

    # Add processing nodes to the graph
    graph.add_node("agent", call_model)  # Main RAG processing step
    graph.add_node("format_output", format_output)  # Output formatting step

    # Define the flow between processing steps
    graph.add_edge("agent", "format_output")  # Agent output flows to formatter

    # Configure graph execution flow
    graph.set_entry_point("agent")  # Start processing with the agent
    graph.set_finish_point("format_output")  # End processing with output formatting

    # Compile and return the executable graph
    return graph.compile(
        checkpointer=MemorySaver()
    )  # Use memory saver for state persistence
