"""
FastAPI application — SHL Assessment Recommender API.

Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → Stateless conversational agent
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import ChatRequest, ChatResponse
from retrieval import build_index
from agent import process_chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load catalog and build FAISS index."""
    logger.info("Starting up — loading catalog and building search index...")
    build_index()
    logger.info("Startup complete — ready to serve requests.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent that recommends SHL assessments based on role requirements.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS for testing from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Readiness check — returns 200 with {"status": "ok"}."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stateless conversational endpoint.

    Takes the full conversation history and returns the next agent reply,
    optionally with a structured shortlist of assessment recommendations.
    """
    logger.info(
        "POST /chat — %d messages, latest: %.100s",
        len(request.messages),
        request.messages[-1].content if request.messages else "(empty)",
    )

    response = await process_chat(request.messages)

    logger.info(
        "Response — %d recommendations, eoc=%s",
        len(response.recommendations),
        response.end_of_conversation,
    )

    return response
