# Golf Equipment Advisor — LangGraph + LlamaIndex

A conversational RAG agent that recommends golf equipment from a curated knowledge base. Built to combine two frameworks that handle different parts of the problem: LangGraph for agent reasoning and tool orchestration, LlamaIndex for retrieval.

## What it does

- Multi-turn chat with persistent memory (LangGraph checkpointing)
- ReAct-style reasoning that decides when to search vs answer from context
- Hybrid retrieval over the golf-fitting corpus (vector + BM25, with reranking)
- LangSmith traces for every run; RAGAS for offline evaluation

## Why both frameworks

Each one is good at half the problem and weak at the other half:

| Capability | LangGraph | LlamaIndex |
|---|---|---|
| Agent reasoning / tool selection | Strong | Limited |
| Multi-step state machines | Strong | Linear only |
| Conversation memory + checkpointing | Strong | None |
| Vector / hybrid retrieval | None | Strong |
| Reranking | Manual | Built-in |
| RAG synthesis | Basic | Optimized |

So LangGraph decides *when* to retrieve and *which* tool to use; LlamaIndex decides *how* to retrieve and *what* to return. Splitting along that line keeps each component focused.

## Flow

```
User query
   │
   ▼
LangGraph agent ──► tool: vector search   ┐
   │                tool: BM25 search      ├─► LlamaIndex retriever
   │                tool: reranker         ┘
   │                                          │
   │◄─────────── retrieved chunks ────────────┘
   │
   ▼
Response (with sources)
   │
   ▼
Memory checkpoint
```

## Stack

LangGraph · LlamaIndex · OpenAI (GPT-4o-mini for reasoning, text-embedding-3-large for embeddings) · LangSmith · RAGAS · Python

## Evaluation

Offline tests run through RAGAS, which scores faithfulness, answer relevancy, and context precision against a labelled question set. LangSmith captures the full trace of each agent run — useful when a retrieval looks weird and you want to see which tool the agent picked, what came back, and how the response was synthesized.

## Key components

- `agent/` — LangGraph state graph and node definitions
- `retrieval/` — LlamaIndex index building, hybrid retriever, LLM reranker
- `eval/` — RAGAS test suite and dataset
- `data/` — golf fitting source documents

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set OpenAI + LangSmith keys
cp .env.example .env

# Build the index (one-time)
python build_index.py

# Run the agent
python run_agent.py
```

## What this project demonstrates

Working with two complementary AI frameworks instead of forcing one to do both jobs. The LangGraph + LlamaIndex split is more code than a single-framework solution, but it makes the agent loop and the retrieval pipeline independently testable, which matters more than line count once a project gets past the demo stage.
