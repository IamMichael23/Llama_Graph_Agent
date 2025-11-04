"""
LLM Agent with Tools

This module demonstrates how to create an LLM agent with custom tools using LangChain and LangGraph.
"""

import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from typing import Optional
from langchain.agents import create_agent
from tools import retrieve_Fitting_Instructions, retrieve_Fitted_Products
from pydantic import SecretStr
import openai
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable

# ============================================================================
# Configure Debug Logging
# ============================================================================
# logging.basicConfig(
#     level=logging.DEBUG,  # Set to DEBUG to see all message flow
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# ============================================================================
# IMPROVEMENT NEEDED: Add imports for memory and logging (2025 best practices)
# ============================================================================
# TODO: Add these imports:
# from langgraph.checkpoint.memory import MemorySaver  # For development
# from langgraph.checkpoint.postgres import PostgresSaver  # For production
# import logging
# from datetime import datetime
#
# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(f'logs/agent_{datetime.now().strftime("%Y%m%d")}.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)
# os.makedirs('logs', exist_ok=True)
# ============================================================================

# ============================================================================
# Load environment variables
# ============================================================================

load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

# ============================================================================
# IMPROVEMENT NEEDED: Replace print with logging
# ============================================================================
# TODO: Replace print statements with logger.info() for better production logging
# logger.info("OPENAI_API_KEY loaded successfully")
# logger.warning("OPENAI_API_BASE not found")
# ============================================================================

if not api_key:
    print("❌ OPENAI_API_KEY not loaded.")
    # TODO: Consider raising an error here instead of continuing
    # raise ValueError("OPENAI_API_KEY is required but not found")
else:
    print("✅ OPENAI_API_KEY loaded.")

if not OPENAI_API_BASE:
    print("⚠️ OPENAI_API_BASE not found.")
else:
    print(f"✅ OPENAI_API_BASE = {OPENAI_API_BASE}")


# ============================================================================
# Initialize Model
# ============================================================================

# ============================================================================
# IMPROVEMENT NEEDED: Add error handling and configuration
# ============================================================================
# TODO: Add error handling for model initialization
# TODO: Add timeout parameter (timeout=30.0)
# TODO: Consider moving config to separate config file
# ============================================================================

llm = ChatOpenAI(
    model='gpt-5-mini',
    temperature=0,  # Good: 0 for consistent, deterministic responses
    base_url=OPENAI_API_BASE,
    api_key=SecretStr(api_key) if api_key else None,
    timeout=1000
)   
print("✅ Model initialized")
# TODO: Replace with logger.info("Model initialized: gpt-5-nano")


# ============================================================================
# Agent
# ============================================================================

# ============================================================================
# IMPROVEMENT NEEDED: Add error handling for file read
# ============================================================================
# TODO: Add try-except for file reading
# try:
#     with open("src/Prompt/golf_advisor_prompt.md", "r", encoding="utf-8") as f:
#         system_message = f.read()
# except FileNotFoundError:
#     logger.error("Prompt file not found")
#     raise
# ============================================================================

# Load custom prompt from markdown file
with open("src/Prompt/golf_advisor_prompt.md", "r", encoding="utf-8") as f:
    system_message = f.read()

# ============================================================================
# 🔴 CRITICAL LIMITATION: Memory disabled!
# ============================================================================
# checkpointer=False means NO conversation memory!
#
# This causes:
# - Agent forgets previous messages in same conversation
# - Can't do multi-turn conversations
# - No error recovery (can't resume from failures)
# - Missing human-in-the-loop capabilities
# - No state inspection for debugging
#
# The comment says "to avoid message ordering issues" but this is likely
# a misunderstanding. Checkpointing is STABLE in LangGraph 2025.
#
# BENEFITS of enabling checkpointing (2025 research):
# ✅ Conversation memory across turns
# ✅ Error recovery (restart from last checkpoint)
# ✅ Human-in-the-loop workflows
# ✅ State inspection for debugging
# ✅ Fault tolerance
#
# TODO: Enable checkpointing:
# 1. Import: from langgraph.checkpoint.memory import MemorySaver
# 2. Create: checkpointer = MemorySaver()
# 3. Use: checkpointer=checkpointer (replace False)
# 4. When invoking, add config with thread_id:
#    config = {"configurable": {"thread_id": "session_001"}}
#    response = agent.invoke({"messages": [...]}, config=config)
#
# For production, use PostgresSaver instead of MemorySaver
# ============================================================================

# Create checkpointer for state management
checkpointer = MemorySaver()

# Create agent with prompt parameter
# Note: Checkpointer ENABLED to maintain proper message history for multi-tool calls
agent = create_agent(
    model=llm,
    tools=[retrieve_Fitting_Instructions], #[retrieved_knowledge_base_product],
    system_prompt=system_message,
    checkpointer=False,  # ✅ Enabled with MemorySaver to maintain message history
    # TODO: Add state_modifier for better control (2025 feature)

)

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # ============================================================================
    # IMPROVEMENT NEEDED: Add comprehensive error handling (2025 best practice)
    # ============================================================================
    # TODO: Wrap entire execution in try-except block
    # TODO: Add logging instead of print statements
    # TODO: Log execution time and token usage
    # ============================================================================

    print("\n" + "="*60)
    print("🔍 DIAGNOSTIC MODE: Tracking all API calls and timings")
    print("="*60)
    # TODO: Replace with logger.info("Starting agent query")

    # Create user query
    user_query = (
        "- Driver Swing Speed: Average 121.5 mph, Peak 124.4 mph\n"
        "- Ball Speed: Up to 180 mph, notable 186 mph\n"
        "- Height: 6 feet 1 inch (185 cm)\n"
        "- Weight: 185 pounds (84 kg)\n"
        "- Age: 49 years old\n"
        "- What should my driver be like?\n"
        "- What Driver should I get?"
    )

    print(f"\n📝 User query length: {len(user_query)} characters")
    

    # Create config with thread_id for checkpointer
    config = {"configurable": {"thread_id": "session_001"}}

    # ============================================================================
    # DIAGNOSTIC: Track total execution time
    # ============================================================================
    print(f"\n⏱️  START TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    start_time = time.time()

    print("\n" + "="*60)
    print("📤 Sending query to agent...")
    print("="*60)

    # ============================================================================
    # DIAGNOSTIC: Comprehensive error handling and timing
    # ============================================================================
    try:
        print("\n🚀 Phase 1: Invoking agent.invoke()...")
        invoke_start = time.time()

        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_query}]},
            config={"configurable": {"thread_id": "session_001"}}
        )

        invoke_end = time.time()
        invoke_duration = invoke_end - invoke_start

        print(f"\n✅ Agent.invoke() completed successfully!")
        print(f"⏱️  Duration: {invoke_duration:.2f} seconds")

    except openai.APITimeoutError as e:
        end_time = time.time()
        duration = end_time - start_time
        print("\n" + "="*60)
        print("❌ TIMEOUT ERROR DETECTED")
        print("="*60)
        print(f"⏱️  Total time before timeout: {duration:.2f} seconds")
        print(f"📝 Timeout setting: 80 seconds")
        print(f"🔍 Error details: {str(e)}")
        print("\n📊 DIAGNOSTIC ANALYSIS:")
        print(f"   - If duration ≈ 80s: Agent planning or synthesis call timed out")
        print(f"   - If duration < 80s: Embedding API call may have timed out")
        print(f"   - Actual duration: {duration:.2f}s")
        print("\n💡 Next steps:")
        print("   1. Check network connectivity to api.agicto.cn")
        print("   2. Verify model 'gpt-5-nano' is valid on your endpoint")
        print("   3. Test endpoint directly with curl/requests")
        print("="*60)
        raise

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print("\n" + "="*60)
        print("❌ UNEXPECTED ERROR")
        print("="*60)
        print(f"⏱️  Time before error: {duration:.2f} seconds")
        print(f"🔍 Error type: {type(e).__name__}")
        print(f"📝 Error message: {str(e)}")
        print("="*60)
        raise

    # ============================================================================
    # DIAGNOSTIC: Track total time
    # ============================================================================
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\n⏱️  END TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print(f"⏱️  TOTAL EXECUTION TIME: {total_duration:.2f} seconds")

    # TODO: Add config parameter when checkpointing is enabled

    print("\n" + "="*60)
    print("Agent Response:")
    print("="*60)

    # ============================================================================
    # IMPROVEMENT NEEDED: Better response parsing and error handling
    # ============================================================================
    # TODO: Add validation that response is not None
    # TODO: Add error handling for malformed responses
    # TODO: Log tool calls for debugging
    # ============================================================================

    if "messages" in response:
        for message in response["messages"]:
            if hasattr(message, 'type') and hasattr(message, 'content'):
                print(f"\n[{message.type.upper()}]: {message.content}")
                # Check for tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        print(f"  [TOOL CALL] {tool_call.get('name', 'unknown')}({tool_call.get('args', {})})")
                        # TODO: Add logger.info(f"Tool called: {tool_call.get('name')}")
    else:
        print(f"\nRaw response: {response}")
        # TODO: Add logger.warning(f"Unexpected response format: {response}")

    print("\n" + "="*60)

    # ============================================================================
    # IMPROVEMENT NEEDED: Add example of multi-turn conversation
    # ============================================================================
    # TODO: Show how to continue conversation with same thread_id:
    # # Follow-up question
    # follow_up = "What about the shaft flex?"
    # response2 = agent.invoke(
    #     {"messages": [("user", follow_up)]},
    #     config={"configurable": {"thread_id": "session_001"}}  # Same thread_id
    # )
    # # Agent will remember previous context!
    # ============================================================================
   