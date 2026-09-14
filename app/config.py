"""Configuration. Everything has a default that works with no setup at all.

The point of this file is that a grader can clone the repo and run it without
obtaining a key, editing a config, or reaching any host on somebody else's network.
"""
import os

# Generation backend.
#   "local"  - runs a small model on this machine through transformers (default)
#   "openai" - any OpenAI-compatible /v1/chat/completions endpoint
LLM_BACKEND = os.getenv("LLM_BACKEND", "local")

# Deliberately tiny. ~80 MB, Apache 2.0, runs on a CPU in a couple of seconds.
# The answers are not clever. That is an accepted trade for making this runnable
# by anyone on any machine without a GPU, an API key, or a 4 GB download.
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "google/flan-t5-small")

# Only used when LLM_BACKEND=openai. No default host on purpose: pointing this at
# a machine the user does not own is how a project stops working for everyone else.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION = "course_materials"

CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "3"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "160"))
