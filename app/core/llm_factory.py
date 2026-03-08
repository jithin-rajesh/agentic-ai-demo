import os
import logging
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

# Configure logging to capture thinking traces
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_factory")


def get_planner_llm() -> ChatNVIDIA:
    """
    Returns the Kimi k2 model instance (via NVIDIA NIM) for reasoning and planning.
    This model has a large context window and strong reasoning capabilities.
    """
    llm = ChatNVIDIA(
        model="moonshotai/kimi-k2-instruct",
        temperature=0.7,
        max_tokens=4096,
    )
    return llm


def get_executor_llm() -> ChatMistralAI:
    """
    Returns the Mistral model instance (via Mistral API directly) for execution.
    This model is fast and optimized for following instructions.
    """
    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
        max_tokens=1024,
    )
    return llm
