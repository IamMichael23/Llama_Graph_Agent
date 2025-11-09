"""
LLM Agent Package

A powerful AI agent with custom tools and memory capabilities.
"""

from .llm_agent import LLMAgent, create_agent
from .tools import get_tools, get_tool_names, ALL_TOOLS
from .memory import get_conversation_memory, get_long_term_memory, ConversationMemory, LongTermMemory
from .embedding import get_embedding_manager, EmbeddingManager

__version__ = "1.0.0"
__author__ = "LLM Agent Team"

__all__ = [
    "LLMAgent",
    "create_agent",
    "get_tools",
    "get_tool_names",
    "ALL_TOOLS",
    "get_conversation_memory",
    "get_long_term_memory",
    "ConversationMemory",
    "LongTermMemory",
    "get_embedding_manager",
    "EmbeddingManager"
]
