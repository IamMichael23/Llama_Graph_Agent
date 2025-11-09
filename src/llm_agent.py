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
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langsmith import traceable
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List, Optional
import json

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
# LLM Debug Callback System
# ============================================================================

class DebugCallback(BaseCallbackHandler):
    """完整的 LLM Debug Callback 系统，捕捉并打印所有内部消息"""
    
    def __init__(self):
        super().__init__()
        self.message_counter = 0
        self.llm_call_counter = 0
        self.current_prompt = None
        self.raw_response_parts = []
    
    def _print_separator(self, title: str = ""):
        """打印分隔线"""
        if title:
            print("\n" + "="*80)
            print(f"  {title}")
            print("="*80)
        else:
            print("-"*80)
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """LLM 调用开始"""
        self.llm_call_counter += 1
        self.raw_response_parts = []
        
        self._print_separator(f"🚀 LLM CALL #{self.llm_call_counter} - START")
        
        # 打印 prompts
        print(f"\n📝 PROMPTS ({len(prompts)} prompts):")
        for i, prompt in enumerate(prompts):
            print(f"\n  Prompt {i+1}:")
            print(f"  {prompt[:500]}{'...' if len(prompt) > 500 else ''}")
        
        # 打印 kwargs 中的有用信息
        if kwargs:
            print(f"\n🔧 KWARGS:")
            for key, value in kwargs.items():
                if key not in ['run_id', 'parent_run_id', 'tags', 'metadata']:
                    print(f"  {key}: {value}")
        
        self.current_prompt = prompts[0] if prompts else None
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """接收到新 token（流式输出）"""
        self.raw_response_parts.append(token)
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 调用结束"""
        self._print_separator(f"✅ LLM CALL #{self.llm_call_counter} - END")
        
        # 打印 kwargs 中的有用信息
        if kwargs:
            print(f"\n🔧 KWARGS:")
            for key, value in kwargs.items():
                if key not in ['run_id', 'parent_run_id', 'tags', 'metadata']:
                    print(f"  {key}: {value}")
        
        # 打印原始响应
        print(f"\n📦 RAW RESPONSE:")
        if response.llm_output:
            print(f"  llm_output: {json.dumps(response.llm_output, indent=2, ensure_ascii=False)}")
        
        # 打印所有 generations
        for i, generation_list in enumerate(response.generations):
            print(f"\n  Generation Batch {i+1}:")
            for j, generation in enumerate(generation_list):
                print(f"\n    Generation {j+1}:")
                
                # 打印消息对象
                if hasattr(generation, 'message'):
                    msg = generation.message
                    print(f"      Message Type: {type(msg).__name__}")
                    print(f"      Content: {str(msg.content)[:500]}")
                    
                    # 检查 tool_calls
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        print(f"\n      🔧 TOOL_CALLS ({len(msg.tool_calls)}):")
                        for k, tc in enumerate(msg.tool_calls):
                            if isinstance(tc, dict):
                                print(f"        [{k}] {json.dumps(tc, indent=6, ensure_ascii=False)}")
                            else:
                                print(f"        [{k}] id={getattr(tc, 'id', 'N/A')}, "
                                      f"name={getattr(tc, 'name', 'N/A')}, "
                                      f"args={getattr(tc, 'args', 'N/A')}")
                    else:
                        print(f"      ⚠️  NO TOOL_CALLS (assistant output without tool_calls)")
                
                # 打印 generation info
                if hasattr(generation, 'generation_info') and generation.generation_info:
                    print(f"\n      Generation Info:")
                    print(f"        {json.dumps(generation.generation_info, indent=6, ensure_ascii=False)}")
        
        # 打印流式收集的原始文本
        if self.raw_response_parts:
            raw_text = ''.join(self.raw_response_parts)
            print(f"\n📄 STREAMED RAW TEXT:")
            print(f"  {raw_text[:1000]}{'...' if len(raw_text) > 1000 else ''}")
        
        self._print_separator()
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """LLM 调用出错"""
        self._print_separator(f"❌ LLM CALL #{self.llm_call_counter} - ERROR")
        print(f"\n  Error Type: {type(error).__name__}")
        print(f"  Error Message: {str(error)}")
        import traceback
        print(f"\n  Traceback:")
        traceback.print_exc()
        self._print_separator()
 
    # def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
    #     """Chain 开始（可能包含 planning 阶段）"""
    #
    #     # ✅ serialized 可能为 None，必须先处理
    #     if serialized is None:
    #         chain_name = "unknown_chain"
    #     else:
    #         # ✅ serialized 里可能没有 name，也可能没有 id，都必须安全处理
    #         chain_name = serialized.get("name")
    #         if chain_name is None:
    #             chain_id = serialized.get("id")
    #             if isinstance(chain_id, list) and chain_id:
    #                 chain_name = chain_id[-1]
    #             else:
    #                 chain_name = str(chain_id) if chain_id else "unknown_chain"
    #
    #     self._print_separator(f"🔗 CHAIN START: {chain_name}")
    #
    #     print("\n📥 INPUTS:")
    #
    #     # ✅ 安全打印 inputs
    #     if inputs and 'messages' in inputs and inputs['messages']:
    #         self._print_messages(inputs['messages'], "Input Messages")
    #     else:
    #         print(json.dumps(inputs, indent=2, ensure_ascii=False))
    # 
    # def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
    #     """Chain 结束"""
    #     if 'messages' in outputs:
    #         self._print_messages(outputs['messages'], "Output Messages")
    # 
    # def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
    #     """Tool 调用开始"""
    #     tool_name = serialized.get('name', 'unknown')
    #     self._print_separator(f"🔧 TOOL CALL: {tool_name}")
    #     print(f"\n  Input: {input_str[:500]}")
    #     
    #     # 打印 kwargs 中的有用信息
    #     if kwargs:
    #         print(f"\n  🔧 KWARGS:")
    #         for key, value in kwargs.items():
    #             if key not in ['run_id', 'parent_run_id', 'tags', 'metadata']:
    #                 print(f"    {key}: {value}")
    # 
    # def on_tool_end(self, output: str, **kwargs: Any) -> None:
    #     """Tool 调用结束"""
    #     print(f"\n  Output: {output[:500]}{'...' if len(output) > 500 else ''}")
    #     self._print_separator()
   
    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Tool 调用出错"""
        print(f"\n  ❌ Tool Error: {type(error).__name__}: {str(error)}")
        self._print_separator()
    
    def _print_messages(self, messages: List[BaseMessage], title: str = "Messages"):
        """打印消息列表的详细信息"""
        print(f"\n📨 {title} ({len(messages)} messages):")
        
        for i, msg in enumerate(messages):
            self.message_counter += 1
            msg_num = self.message_counter
            
            print(f"\n  ┌─ Message #{msg_num} (Index {i}) ──────────────────────────────")
            
            # 消息类型和角色
            msg_type = getattr(msg, 'type', None) or getattr(msg, 'role', None) or type(msg).__name__
            print(f"  │ Type/Role: {msg_type}")
            
            # 内容 - 显示完整内容，但每行限制长度
            if hasattr(msg, 'content'):
                content = msg.content
                if content:
                    content_str = str(content)
                    # 如果内容太长，显示前1000个字符，并提示总长度
                    if len(content_str) > 1000:
                        print(f"  │ Content (showing first 1000 of {len(content_str)} chars):")
                        print(f"  │ {content_str[:1000]}...")
                        print(f"  │ ... ({len(content_str) - 1000} more characters)")
                    else:
                        # 内容不太长时，完整显示，但每行限制长度以便阅读
                        lines = content_str.split('\n')
                        if len(lines) == 1:
                            # 单行但可能很长
                            if len(content_str) > 500:
                                print(f"  │ Content ({len(content_str)} chars, showing first 500):")
                                print(f"  │ {content_str[:500]}...")
                                print(f"  │ ... ({len(content_str) - 500} more characters)")
                            else:
                                print(f"  │ Content: {content_str}")
                        else:
                            print(f"  │ Content ({len(lines)} lines, {len(content_str)} chars):")
                            # 显示所有行，但每行限制长度
                            for line_idx, line in enumerate(lines[:50]):  # 最多显示前50行
                                if len(line) > 200:
                                    print(f"  │   [{line_idx+1:3d}] {line[:200]}... ({len(line)} chars)")
                                else:
                                    print(f"  │   [{line_idx+1:3d}] {line}")
                            if len(lines) > 50:
                                print(f"  │   ... ({len(lines) - 50} more lines not shown)")
                else:
                    print(f"  │ Content: (empty)")
            
            # tool_calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"  │")
                print(f"  │ 🔧 TOOL_CALLS ({len(msg.tool_calls)}):")
                for j, tc in enumerate(msg.tool_calls):
                    if isinstance(tc, dict):
                        tc_id = tc.get('id', 'N/A')
                        tc_name = tc.get('name', 'N/A')
                        tc_args = tc.get('arguments', 'N/A')
                    else:
                        tc_id = getattr(tc, 'id', 'N/A')
                        tc_name = getattr(tc, 'name', 'N/A')
                        tc_args = getattr(tc, 'args', 'N/A')
                    
                    print(f"  │   [{j}] id={tc_id}, name={tc_name}")
                    if tc_args and tc_args != 'N/A':
                        args_str = str(tc_args) if not isinstance(tc_args, dict) else json.dumps(tc_args, ensure_ascii=False)
                        print(f"  │       args: {args_str[:150]}{'...' if len(args_str) > 150 else ''}")
            else:
                if msg_type in ['ai', 'assistant']:
                    print(f"  │")
                    print(f"  │ ⚠️  NO TOOL_CALLS (assistant output without tool_calls)")
            
            # tool_call_id (对于 tool 消息)
            if hasattr(msg, 'tool_call_id'):
                tool_call_id = getattr(msg, 'tool_call_id', None)
                if tool_call_id:
                    print(f"  │")
                    print(f"  │ 🔗 tool_call_id: {tool_call_id}")
                else:
                    print(f"  │")
                    print(f"  │ ⚠️  MISSING tool_call_id (tool message without tool_call_id)")
            
            # 其他属性
            if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                print(f"  │")
                print(f"  │ Additional kwargs: {json.dumps(msg.additional_kwargs, indent=4, ensure_ascii=False)[:200]}")
            
            print(f"  └─────────────────────────────────────────────────────────")

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

# Load custom prompt from markdown file
with open("src/Prompt/golf_advisor_prompt.md", "r", encoding="utf-8") as f:
    system_message = f.read()

llm = ChatOpenAI(
    model='gpt-5',
    temperature=0,  # Good: 0 for consistent, deterministic responses
    base_url=OPENAI_API_BASE,
    api_key=SecretStr(api_key) if api_key else None,
    timeout=1000
)

# 添加 Debug Callback
debug_callback = DebugCallback()
llm = llm.with_config({"callbacks": [debug_callback]})

# ============================================================================
# ✅ PATCH: Print Final OpenAI API Payload (REAL messages sent to LLM)
# ============================================================================

from openai.resources.chat.completions import Completions
_original_create = Completions.create

def patched_create(self, *args, **kwargs):
    print("\n" + "="*80)
    print("📨 FINAL API PAYLOAD SENT TO OPENAI")
    print("="*80)

    try:
        import json
        print(json.dumps(kwargs, indent=2, ensure_ascii=False))
    except Exception:
        print(kwargs)

    print("="*80)
    print("✅ END LLM PAYLOAD\n")

    return _original_create(self, *args, **kwargs)

Completions.create = patched_create

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

# ============================================================================

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
checkpointer.storage.clear()   # Ste 强制清空旧 memory

# Create agent with prompt parameter
# Note: Checkpointer ENABLED to maintain proper message history for multi-tool calls

# ============================================================================
# DEBUG: 检查工具注册信息（非常关键）
# ============================================================================
print("=== DEBUG TOOL REGISTRATION ===")
for tool in [retrieve_Fitting_Instructions, retrieve_Fitted_Products]:
    print("Tool object:", tool)
    print("Tool name:", tool.name)
    print("Tool description:", tool.description)
    print("----")
# ============================================================================

# Create agent executor using LangGraph's recommended pattern
agent = create_agent(
    model=llm,
    tools=[retrieve_Fitting_Instructions],
    system_prompt=system_message,  # Note: create_react_agent uses 'prompt' not 'system_prompt'
    checkpointer=checkpointer
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
    thread_id = "937"
    print(f"📋 Using thread_id: {thread_id}")

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
            {
                "messages": [
                    {"role": "user", "content": user_query} 
                ]
            },
            config={
                "configurable": {"thread_id": thread_id},
                "callbacks": [debug_callback]  # ✅ ✅ ✅ 关键
            }  # Ste 必须使用config来传递thread_id和debug_callback
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
    print("📤 AGENT RESPONSE")
    print("="*60)
    
    # Extract final response from agent executor output
    if isinstance(response, dict):
        if "messages" in response:
            final_message = response["messages"][-1]
            if hasattr(final_message, 'content'):
                print(f"\n💬 Final Response: {final_message.content}")
            else:
                print(f"\n💬 Final Response: {final_message}")
        else:
            print(f"\n💬 Response: {response}")
    else:
        print(f"\n💬 Response: {response}")
    
    print("="*60)
   

    # ============================================================================
    # IMPROVEMENT NEEDED: Better response parsing and error handling
    # ============================================================================
    # TODO: Add validation that response is not None
    # TODO: Add error handling for malformed responses
    # TODO: Log tool calls for debugging
    # ============================================================================

    

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
   