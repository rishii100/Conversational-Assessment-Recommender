"""
Configuration module — loads environment variables from .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = "llama-3.1-8b-instant"
TOP_K_RETRIEVAL: int = 8
