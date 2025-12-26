# 🏌️ Golf Equipment AI Advisor

**An intelligent conversational agent built with LangGraph and LlamaIndex for personalized golf equipment recommendations**

This project demonstrates advanced AI agent development by combining two complementary frameworks to create a production-ready RAG (Retrieval-Augmented Generation) system with multi-turn reasoning capabilities.

---

## 🎯 Project Overview

An intelligent golf equipment advisor that provides expert recommendations through:
- **Multi-turn conversational reasoning** with context-aware responses
- **Retrieval-Augmented Generation (RAG)** over curated domain knowledge
- **Automated evaluation** using industry-standard metrics (LangSmith, RAGAS)
- **Tool orchestration** for dynamic information retrieval

---

## 🤔 Why LangGraph + LlamaIndex?

### The Perfect Division of Labor

Each framework excels at what it's designed for:

| Capability | LangGraph | LlamaIndex |
|------------|-----------|------------|
| **Agent Reasoning** | ✅ Excellent | ⚠️ Limited |
| **Tool Orchestration** | ✅ Built-in | ⚠️ Basic |
| **Multi-step Workflows** | ✅ State machines | ⚠️ Linear only |
| **Conversation Memory** | ✅ Checkpointing | ❌ None |
| **Document Retrieval** | ❌ None | ✅ Excellent |
| **Vector Search** | ❌ None | ✅ Built-in |
| **Reranking** | ❌ Manual | ✅ Built-in |
| **RAG Synthesis** | ⚠️ Basic | ✅ Optimized |

### 🎯 The Synergy

```
LangGraph decides:           LlamaIndex executes:
├─ WHEN to retrieve          ├─ HOW to retrieve
├─ WHICH tool to use         ├─ WHERE to search
├─ WHETHER to iterate        ├─ WHAT to return
└─ HOW to respond            └─ WHY it's relevant

Together = Intelligent + Accurate
```

**Example Flow:**

1. **LangGraph**: "User asks about products. I should search the knowledge base."
2. **LlamaIndex**: Searches documents, finds top 3 relevant chunks
3. **LangGraph**: "Results look good. I'll format a personalized answer."
4. **Result**: Accurate, source-backed recommendation

---

## 🛠️ Tech Stack

- **LangGraph**: Agent orchestration, state management, multi-turn reasoning
- **LlamaIndex**: Vector embeddings, document retrieval, hybrid search, reranking
- **OpenAI API**: GPT-4o-mini for reasoning, text-embedding-3-large for embeddings
- **LangSmith**: Agent evaluation and observability
- **RAGAS**: Automated RAG quality assessment
- **Python**: Core implementation

---

## 🎯 Key Features

- ✅ **Multi-turn conversations** with persistent memory and context
- ✅ **ReAct-style reasoning** with dynamic tool selection
- ✅ **Hybrid retrieval** combining vector search and BM25
- ✅ **LLM-based reranking** for improved relevance
- ✅ **Automated evaluation** with comprehensive test suites
- ✅ **Production-ready** architecture with error handling

---

## 🧠 Architecture

```
User Query → LangGraph Agent → Tool Selection → LlamaIndex RAG → Response
                ↓
         Memory Checkpointing
                ↓
         Context-Aware Reasoning
```

**Key Components:**
- Agent orchestration layer (LangGraph)
- RAG retrieval system (LlamaIndex)
- Vector store with embeddings
- Evaluation framework (LangSmith + RAGAS)

---

## 📊 Evaluation & Quality Assurance

- **RAGAS Metrics**: Measures retrieval quality, answer relevance, and faithfulness

---

## 🎓 Learning Outcomes

This project demonstrates:
- **Framework integration**: Combining complementary AI frameworks effectively
- **RAG implementation**: Building production-ready retrieval systems
- **Agent architecture**: Designing multi-turn conversational AI
- **Evaluation practices**: Implementing comprehensive quality assurance
- **Technical decision-making**: Understanding trade-offs between frameworks

---

**Built with LangGraph + LlamaIndex**

