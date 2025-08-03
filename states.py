from typing import Annotated, List, TypedDict

from langchain_core.messages import (
    BaseMessage,
)
from langgraph.graph import add_messages


class InputState(TypedDict):
    """Graph Input"""

    query: str
    role: str  # user, assistant


class OutputState(TypedDict):
    """Graph Output"""

    answer: str


class OverallState(InputState, OutputState):
    """Overall State"""

    messages: Annotated[List[BaseMessage], add_messages]
