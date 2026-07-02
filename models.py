"""
Pydantic models for the API request/response schema.

The schema is non-negotiable per the assignment spec:
- POST /chat receives a list of messages
- Response contains reply, recommendations (list or empty), and end_of_conversation flag
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChatMessage(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="The message content")


class ChatRequest(BaseModel):
    """Request body for POST /chat — stateless, carries full conversation history."""
    messages: list[ChatMessage] = Field(
        ..., description="Full conversation history as a list of messages"
    )


class Recommendation(BaseModel):
    """A single assessment recommendation from the SHL catalog."""
    name: str = Field(..., description="Assessment name from the catalog")
    url: str = Field(..., description="Catalog URL for the assessment")
    test_type: str = Field(..., description="Test type code(s), e.g. 'K', 'P', 'A,S'")


class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    reply: str = Field(..., description="The agent's natural language reply")
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="Empty when gathering context; 1-10 items when committing to a shortlist",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the agent considers the task complete",
    )
