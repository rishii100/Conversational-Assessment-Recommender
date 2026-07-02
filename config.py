"""
Configuration module — loads environment variables from .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-2.0-flash"
TOP_K_RETRIEVAL: int = 25
