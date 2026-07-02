"""
Configuration module — loads environment variables from .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "25"))
