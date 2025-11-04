# Timeout Diagnostic Guide

This guide explains how to use the diagnostic tools to identify and fix the timeout issue.

## Quick Start

### Step 1: Test the Endpoint

First, test if your custom API endpoint is working:

```bash
cd /Users/blackchina23/Fuck_School/Western_MDA/CS9146_Artifical_Intelligence/Agent/Agent
python src/test_endpoint.py
```

**What to look for:**
- ✅ All tests pass: Endpoint is working, issue is elsewhere
- ❌ Network test fails: Check internet/firewall/URL
- ❌ LLM test times out: **This is your problem!**
- ❌ LLM test fails with 400/404: Model name invalid
- ❌ Embedding test times out: Add timeout to embedding model

### Step 2: Run the Agent with Diagnostics

Run your agent with the new diagnostic logging:

```bash
cd /Users/blackchina23/Fuck_School/Western_MDA/CS9146_Artifical_Intelligence/Agent/Agent
python src/llm_agent.py
```

**The output will show:**

```
============================================================
🔍 DIAGNOSTIC MODE: Tracking all API calls and timings
============================================================

⏱️  START TIME: 2025-11-03 14:23:45.123

============================================================
📤 Sending query to agent...
============================================================

🚀 Phase 1: Invoking agent.invoke()...

   [Then you'll see tool invocations...]

============================================================
🎯 TOOL INVOKED: retrieved_knowledge_base_product
📝 Query: [your query]
⏱️  Tool start time: 14:23:45.234
============================================================

   [Then embedding loader diagnostics...]

⏱️  [load_embedding_index] START: 14:23:45.345
📂 Loading index from disk...
📦 Index successfully unpacked from knowledge_base
⏱️  Load duration: 2.34 seconds

   [Then retrieval phases...]

📊 Phase 1: Loading index from storage...
📊 Phase 2: Setting up hybrid retriever (Vector + BM25)...
⏱️  Retriever setup: 0.12 seconds

📊 Phase 3: Retrieving relevant nodes...
🚨 WARNING: This phase makes an API call to embed the query!
   Endpoint: https://api.agicto.cn/v1
   Model: text-embedding-3-small
⏱️  Retrieval duration: 1.23 seconds  <-- Watch this time!
📦 Retrieved 5 nodes

   [And so on...]
```

---

## Understanding the Output

### Normal Execution Times

| Phase | Normal Time | Warning if > | Critical if > |
|-------|-------------|--------------|---------------|
| Load Index | 2-5s | 10s | 20s |
| Retriever Setup | 0.1-0.5s | 2s | 5s |
| Query Embedding | 1-3s | 10s | 30s |
| Agent Planning | 3-8s | 30s | 60s |
| Agent Synthesis | 5-12s | 40s | 70s |
| **Total** | **11-28s** | **60s** | **90s** |

### Timeout Scenarios

#### Scenario 1: Timeout at Query Embedding

```
📊 Phase 3: Retrieving relevant nodes...
🚨 WARNING: This phase makes an API call to embed the query!
   Endpoint: https://api.agicto.cn/v1
   Model: text-embedding-3-small
[hangs here for 30+ seconds]
❌ TIMEOUT ERROR DETECTED
⏱️  Total time before timeout: 31.45 seconds
```

**Diagnosis:** Embedding API is timing out
**Fix:** Add `timeout=30.0` to OpenAIEmbedding in `embedding_loader.py` line 58-62

#### Scenario 2: Timeout at Agent Planning

```
🚀 Phase 1: Invoking agent.invoke()...
[hangs for 80 seconds]
❌ TIMEOUT ERROR DETECTED
⏱️  Total time before timeout: 80.12 seconds
📊 DIAGNOSTIC ANALYSIS:
   - If duration ≈ 80s: Agent planning or synthesis call timed out
```

**Diagnosis:** LLM endpoint (gpt-5-nano) is timing out
**Fix:**
1. Test with `test_endpoint.py` to confirm
2. Increase timeout to 120s
3. Or switch to official OpenAI endpoint

#### Scenario 3: Timeout at Agent Synthesis

```
✅ TOOL COMPLETED: retrieved_knowledge_base_product
📦 Returned 5 product nodes
⏱️  Tool duration: 8.23 seconds
[hangs for 80 seconds]
❌ TIMEOUT ERROR DETECTED
⏱️  Total time before timeout: 82.34 seconds
```

**Diagnosis:** Second LLM call (synthesis) timing out
**Root Cause:** Usually means endpoint is slow or under load
**Fix:** Same as Scenario 2

---

## Diagnostic Checklist

Use this checklist to systematically debug the issue:

### [ ] 1. Endpoint Connectivity Test

```bash
python src/test_endpoint.py
```

**Results:**
- [ ] Network test passed
- [ ] LLM test passed
- [ ] Embedding test passed
- [ ] All response times < 10 seconds

**If any fail:** Follow recommendations in test output

### [ ] 2. Check Environment Variables

```bash
# In your terminal:
echo $OPENAI_API_BASE
echo $OPENAI_API_KEY  # Should show masked key
echo $EMBEDDING_KEY    # Should show masked key
```

**Verify:**
- [ ] OPENAI_API_BASE = https://api.agicto.cn/v1
- [ ] OPENAI_API_KEY is set
- [ ] EMBEDDING_KEY is set
- [ ] Both keys are valid

### [ ] 3. Run Agent with Diagnostics

```bash
python src/llm_agent.py 2>&1 | tee diagnostic_log.txt
```

**This will:**
- Run the agent
- Show all diagnostic output
- Save output to `diagnostic_log.txt` for analysis

### [ ] 4. Analyze the Output

**Find where it times out:**
- [ ] Timeout at "Phase 3: Retrieving" → Embedding API issue
- [ ] Timeout at "agent.invoke()" → LLM API issue
- [ ] Timeout after "TOOL COMPLETED" → LLM synthesis issue

### [ ] 5. Identify the Root Cause

Based on diagnostic output, identify:

**A. Network/Endpoint Issues:**
- Endpoint unreachable
- Very high latency (>10s per call)
- Intermittent connectivity

**B. Model Issues:**
- Model name 'gpt-5-nano' not recognized
- Model taking too long to respond

**C. Configuration Issues:**
- Timeout too short
- Missing timeout on embedding API

**D. Load/Rate Limiting:**
- Endpoint slow during certain times
- Rate limiting activated

---

## Common Fixes

### Fix 1: Add Embedding Timeout

**File:** `src/embedding_loader.py` line 58-62

```python
embed_model=OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("EMBEDDING_KEY"),
    api_base=os.getenv("OPENAI_API_BASE"),
    timeout=30.0  # ADD THIS LINE
)
```

### Fix 2: Increase Agent LLM Timeout

**File:** `src/llm_agent.py` line 79-85

```python
llm = ChatOpenAI(
    model='gpt-5-nano',
    temperature=0,
    base_url=OPENAI_API_BASE,
    api_key=SecretStr(api_key) if api_key else None,
    timeout=120,  # INCREASE from 80 to 120
)
```

### Fix 3: Try Different Model

**File:** `src/llm_agent.py` line 80

```python
# If 'gpt-5-nano' is invalid, try:
model='gpt-4o-mini',  # Or 'gpt-3.5-turbo'
```

### Fix 4: Switch to Official OpenAI Endpoint

**File:** `.env`

```bash
# Replace:
OPENAI_API_BASE=https://api.agicto.cn/v1

# With:
OPENAI_API_BASE=https://api.openai.com/v1
```

**File:** `src/llm_agent.py` line 80

```python
# Use standard model:
model='gpt-4o-mini',  # or 'gpt-3.5-turbo'
```

### Fix 5: Fix Embedding Model Mismatch

**File:** `src/embedding_loader.py` line 50

```python
# Change from:
model="text-embedding-3-small",

# To match embedding.py:
model="text-embedding-3-large",
```

---

## Expected Diagnostic Output (Success Case)

When everything works, you'll see:

```
============================================================
🔍 DIAGNOSTIC MODE: Tracking all API calls and timings
============================================================

⏱️  START TIME: 2025-11-03 14:23:45.123

🚀 Phase 1: Invoking agent.invoke()...

============================================================
🎯 TOOL INVOKED: retrieved_knowledge_base_product
⏱️  Tool start time: 14:23:45.456
============================================================

⏱️  [load_embedding_index] START: 14:23:45.567
📂 Loading index from disk...
⏱️  Load duration: 2.34 seconds

📊 Phase 2: Setting up hybrid retriever...
⏱️  Retriever setup: 0.12 seconds

📊 Phase 3: Retrieving relevant nodes...
⏱️  Retrieval duration: 1.23 seconds  ✅ < 3s
📦 Retrieved 5 nodes

📊 Phase 4: Post-processing...
⏱️  Post-processing: 0.05 seconds
📦 Filtered to 3 nodes

⏱️  [read_and_retrieve] TOTAL: 3.74 seconds  ✅ < 10s

============================================================
✅ TOOL COMPLETED: retrieved_knowledge_base_product
📦 Returned 3 product nodes
⏱️  Tool duration: 3.74 seconds  ✅ < 10s
============================================================

✅ Agent.invoke() completed successfully!
⏱️  Duration: 12.45 seconds  ✅ < 30s

⏱️  TOTAL EXECUTION TIME: 12.45 seconds  ✅ < 60s

[Response output...]
```

**All times are normal! ✅**

---

## Expected Diagnostic Output (Timeout Case)

When timing out, you'll see:

```
============================================================
🔍 DIAGNOSTIC MODE: Tracking all API calls and timings
============================================================

⏱️  START TIME: 2025-11-03 14:23:45.123

🚀 Phase 1: Invoking agent.invoke()...

============================================================
🎯 TOOL INVOKED: retrieved_knowledge_base_product
⏱️  Tool start time: 14:23:45.456
============================================================

⏱️  [load_embedding_index] START: 14:23:45.567
📂 Loading index from disk...
⏱️  Load duration: 2.34 seconds

📊 Phase 2: Setting up hybrid retriever...
⏱️  Retriever setup: 0.12 seconds

📊 Phase 3: Retrieving relevant nodes...
🚨 WARNING: This phase makes an API call to embed the query!
   Endpoint: https://api.agicto.cn/v1
   Model: text-embedding-3-small

[HANGS HERE... 80+ seconds pass]

============================================================
❌ TIMEOUT ERROR DETECTED
============================================================
⏱️  Total time before timeout: 82.67 seconds  ❌
📝 Timeout setting: 80 seconds
🔍 Error details: Request timed out.

📊 DIAGNOSTIC ANALYSIS:
   - Actual duration: 82.67s
   - Duration > 80s: Agent planning or synthesis call timed out

💡 Next steps:
   1. Check network connectivity to api.agicto.cn
   2. Verify model 'gpt-5-nano' is valid on your endpoint
   3. Test endpoint directly with curl/requests
============================================================
```

**The timeout occurred after 82.67 seconds, which is just over the 80s timeout!**
**This suggests the LLM call is hanging.**

---

## Troubleshooting Decision Tree

```
Start
  |
  ├─ Run test_endpoint.py
  |    |
  |    ├─ Network test fails?
  |    |    └─> Check internet/firewall/URL
  |    |
  |    ├─ LLM test times out?
  |    |    └─> Endpoint is slow → Increase timeout or switch endpoint
  |    |
  |    ├─ LLM test fails 400/404?
  |    |    └─> Model invalid → Change to gpt-4o-mini
  |    |
  |    └─ All pass?
  |         └─> Continue to agent test
  |
  └─ Run llm_agent.py with diagnostics
       |
       ├─ Timeout at "Phase 3: Retrieving"?
       |    └─> Add timeout to embedding model
       |
       ├─ Timeout at agent.invoke()?
       |    └─> LLM endpoint slow → Increase timeout
       |
       ├─ Timeout after tool completes?
       |    └─> Synthesis call slow → Increase timeout
       |
       └─ Success?
            └─> No issues! Context was not the problem.
```

---

## Next Steps

1. **Run endpoint test:** `python src/test_endpoint.py`
2. **Run agent with diagnostics:** `python src/llm_agent.py`
3. **Analyze output:** Use this guide to identify issue
4. **Apply fix:** Follow Common Fixes section
5. **Test again:** Verify fix works
6. **Report findings:** Share diagnostic output if issue persists

---

## Contact/Support

If you've followed this guide and still have issues:

1. Save diagnostic output: `python src/llm_agent.py &> full_diagnostic.log`
2. Run endpoint test: `python src/test_endpoint.py &> endpoint_test.log`
3. Share both logs with your support team
4. Include your `.env` configuration (redact API keys!)

Good luck! 🚀
