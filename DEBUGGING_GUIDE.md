# LangGraph Tool Message Error - Debugging Guide

## Error Fixed
```
ValueError: {'message': "Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.", 'type': 'invalid_request_error', 'param': 'messages.[2].role', 'code': None}
```

## Root Cause
The agent had `checkpointer=False` which disabled state persistence, causing message history corruption during tool calls. Without proper checkpointing, the sequence of:
1. User message
2. Assistant message with tool_calls
3. Tool response messages

...would become malformed, violating OpenAI's strict message ordering requirements.

## Fixes Applied

### 1. Enabled Checkpointer ✅
**File:** `src/llm_agent.py:168`
**Change:** `checkpointer=False` → `checkpointer=checkpointer`

This enables proper state persistence so tool calls and responses stay synchronized.

### 2. Added Config with Thread ID ✅
**File:** `src/llm_agent.py:205-227`

```python
config = {"configurable": {"thread_id": "session_001"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": user_query}]},
    config=config
)
```

The thread_id allows the checkpointer to track conversation state.

### 3. Disabled Parallel Tool Calls ✅
**File:** `src/llm_agent.py:101`

```python
llm = ChatOpenAI(
    ...
    parallel_tool_calls=False,
)
```

This prevents race conditions when multiple tools are called simultaneously.

### 4. Updated langgraph.json ✅
**File:** `langgraph.json`

Now properly configured for LangGraph Studio with:
- Descriptive name: "golf-advisor-agent"
- Graph name: "golf_advisor"
- Metadata with description and version

### 5. Added Message Validation Helper ✅
**File:** `src/llm_agent.py:131-171`

A `validate_message_sequence()` function that catches invalid message ordering before it reaches the API.

## Testing the Fix

### Test 1: Run the Agent
```bash
cd /Users/blackchina23/Fuck_School/Western_MDA/CS9146_Artifical_Intelligence/Agent/Agent
.venv/bin/python src/llm_agent.py
```

**Expected:** No more "invalid message role" errors. The agent should successfully call `retrieve_Fitting_Instructions` and generate a response.

### Test 2: Verify State Persistence
```python
# After running the agent, check if state was saved:
from langgraph.checkpoint.memory import MemorySaver

config = {"configurable": {"thread_id": "session_001"}}
state = agent.get_state(config)
print(f"Messages in state: {len(state.values.get('messages', []))}")
```

### Test 3: Multi-Turn Conversation
```python
# First turn
config = {"configurable": {"thread_id": "test_002"}}
response1 = agent.invoke(
    {"messages": [{"role": "user", "content": "What driver for 121 mph swing?"}]},
    config=config
)

# Second turn (should remember context)
response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What about the shaft?"}]},
    config=config
)
```

## Using LangGraph Studio for Debugging

### Start LangGraph Studio
```bash
cd /Users/blackchina23/Fuck_School/Western_MDA/CS9146_Artifical_Intelligence/Agent/Agent
langgraph dev
```

Then open: http://127.0.0.1:2024

### Features Available:
1. **Visual Graph Inspection** - See your agent's execution flow
2. **Time-Travel Debugging** - Rewind to any checkpoint and inspect state
3. **Message History Viewer** - See exact message sequence at each step
4. **State Inspection** - View all state variables
5. **Manual Intervention** - Add corrective messages if needed

### Debug a Failed Run:
1. Run your agent in Studio
2. When error occurs, click on the failed node
3. Inspect the messages array at that point
4. Look for:
   - Missing tool_call_id matches
   - Tool messages without preceding assistant messages
   - Extra system messages inserted

## Common Issues and Solutions

### Issue: "Service load is too high"
**Cause:** API server overloaded
**Solution:** Add retry logic or wait and try again

### Issue: QueryFusionRetriever still causing problems
**Cause:** Internal LLM calls from retriever interfering with agent
**Solution:** Disable query fusion or use simpler retriever:

```python
# In embedding_loader.py, replace fusion retriever with:
retriever = index.as_retriever(similarity_top_k=10)
nodes = retriever.retrieve(user_query)
```

### Issue: Still getting tool message errors
**Cause:** Might be resuming interrupted conversation incorrectly
**Solution:** Use a fresh thread_id or clear checkpoints:

```python
# Use new thread
config = {"configurable": {"thread_id": f"session_{datetime.now().timestamp()}"}}
```

## Additional Resources

### OpenAI Message Format Requirements
- Tool messages MUST have `tool_call_id` matching an `id` from assistant's `tool_calls`
- Tool messages MUST immediately follow the assistant message containing tool_calls
- No other messages can be inserted between them

### LangGraph Documentation
- Checkpointing: https://docs.langchain.com/oss/python/langgraph/interrupts
- Message Management: https://langchain-ai.github.io/langgraphjs/how-tos/manage-conversation-history/
- INVALID_CHAT_HISTORY Error: https://langchain-ai.github.io/langgraph/troubleshooting/errors/INVALID_CHAT_HISTORY/

### GitHub Issues Referenced
- [#544 - CheckPointer makes tool call break](https://github.com/langchain-ai/langgraph/discussions/544)
- [#1398 - Two tool calls error](https://github.com/langchain-ai/langgraph/discussions/1398)
- [#4341 - Resuming interrupted tool issue](https://github.com/langchain-ai/langgraph/discussions/4341)

## Success Metrics

After these fixes, you should see:
- ✅ No "invalid message role" errors
- ✅ Tools execute successfully
- ✅ Agent maintains conversation memory
- ✅ State persists across invocations
- ✅ Clean message sequences in LangGraph Studio

## Next Steps

1. **Test the agent** with the fixes applied
2. **Open LangGraph Studio** to visually debug any remaining issues
3. **Enable LangSmith tracing** for production monitoring
4. **Consider adding** more robust error handling and retry logic

Good luck debugging! 🎯
