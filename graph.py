""" " langgraph implementations for a rag system with."""

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph

from llm import get_llm
from query import query_vector_store
from states import InputState, OutputState, OverallState


def create_rag_agent():
    """RAG Agent for QA"""

    """ llm with tools """
    llm = get_llm()

    def call_model(state: OverallState):
        """invoke llm"""
        local_messages = state.get("messages", [])
        human_message = HumanMessage(content=state["query"])
        local_messages.append(human_message)

        context = query_vector_store(state["query"])

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

        response = llm.invoke([system_message] + local_messages)

        state["messages"].append(response)

        return state

    def format_output(state: OverallState):
        """Format the output"""
        state["answer"] = state["messages"][-1].content
        return state

    """ define the graph """
    graph = StateGraph(OverallState, input=InputState, output=OutputState)
    graph.add_node("agent", call_model)
    graph.add_node("format_output", format_output)
    graph.add_edge("agent", "format_output")

    """ define the transitions """
    graph.set_entry_point("agent")
    graph.set_finish_point("format_output")

    """ return the compiled graph """
    return graph.compile()
