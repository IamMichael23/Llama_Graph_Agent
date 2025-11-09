"""Load API keys from environment file."""
import os
from pathlib import Path
from dotenv import load_dotenv


def load_key():
    """Load API keys from .env file in the project root."""
    # Get the directory where this file is located (src/config/)
    current_dir = Path(__file__).parent
    # Navigate to project root (two levels up from src/config/)
    env_path = current_dir.parent.parent / ".env"

    # Load environment variables from .env file
    load_dotenv(env_path)

    # Verify at least OpenAI keys are loaded
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    return True
