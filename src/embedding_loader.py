import warnings
import os
import logging
import time
from datetime import datetime

from config.load_key import load_key
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex 
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import StorageContext,load_index_from_storage
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from langsmith import traceable
from langsmith.wrappers import wrap_openai

def load_embedding_index(path: str = "src/storage/products_emb/"):
    # ============================================================================
    # DIAGNOSTIC: Track index loading time
    # ============================================================================
    print(f"\n⏱️  [load_embedding_index] START: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    load_start = time.time()

    # ============================================================================
    # IMPROVEMENT NEEDED: Add error handling
    # ============================================================================
    # TODO: Add try-except to handle missing storage directory or corrupted index
    # TODO: Add @retry decorator for API failures
    # ============================================================================

    if "EMBEDDING_KEY" not in os.environ:
        load_key()
    # Load index from storage without recomputing embeddings

    # ============================================================================
    # 🔴 CRITICAL BUG: Model mismatch!
    # ============================================================================
    # Line 95 (create function) uses: text-embedding-3-large
    # Line 130 (load function) uses:  text-embedding-3-small  ❌ WRONG!
    #
    # MUST BE CONSISTENT or retrieval will fail!
    #
    # TODO: Change text-embedding-3-small → text-embedding-3-large
    #
    # Research (2025): text-embedding-3-large achieves 80.5% accuracy
    #                  text-embedding-3-small achieves 75.8% accuracy
    #                  Cost: $0.13/M vs $0.02/M tokens (worth it for accuracy)
    # ============================================================================

    print("📂 Loading index from disk...")
    storage_context = StorageContext.from_defaults(persist_dir=path)
    index = load_index_from_storage(storage_context,
                                    embed_model=OpenAIEmbedding(
                                        model = "text-embedding-3-small",
                                        api_key=os.getenv("EMBEDDING_KEY"),
                                        api_base=os.getenv("OPENAI_API_BASE")),
                                        index_cls=VectorStoreIndex)

    load_end = time.time()
    load_duration = load_end - load_start

    print("\n" + "="*60)
    print("📦 Index successfully unpacked from knowledge_base")
    print(f"⏱️  Load duration: {load_duration:.2f} seconds")
    print("="*60)

    return index


# ✅ 彻底关闭全局默认 LLM

def read_and_query(user_query: str = "what do we have?"):
    # ============================================================================
    # IMPROVEMENT NEEDED: Add error handling with retries (2025 best practice)
    # ============================================================================
    # TODO: Add @retry decorator:
    # @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    #
    # TODO: Wrap in try-except:
    # try:
    #     ... function logic ...
    # except Exception as e:
    #     logger.error(f"Error in read_and_query: {str(e)}")
    #     raise
    # ============================================================================

    # ============================================================================
    # IMPROVEMENT NEEDED: Add configuration parameters
    # ============================================================================
    # TODO: Add parameters: similarity_top_k=5, response_mode="compact"
    # Retrieving more chunks (5 vs default 2) improves answer quality
    # ============================================================================

    if "EMBEDDING_KEY" not in os.environ:
        load_key()
    index = load_embedding_index()
    query_engine = index.as_query_engine(
        streaming=False,
        # TODO: Add similarity_top_k=5 for better retrieval
        llm=OpenAILike(
            model="gpt-5-nano",
            api_base=os.getenv("OPENAI_API_BASE"),
            api_key = os.getenv("OPENAI_API_KEY"),
            is_chat_model=True
            # TODO: Add timeout=30.0 to prevent hanging

            ))

    print("\n" + "="*60)
    print("Dont BB I am Thinking ...")
    print("="*60)

    # ============================================================================
    # IMPROVEMENT NEEDED: Add response validation
    # ============================================================================
    # TODO: Check if response is None or empty before returning
    # ============================================================================

    response = query_engine.query(user_query)
    return response


@traceable
def read_and_retrieve(user_query: str = "what do we have?", doc_path: str = "src/storage/products_emb/"):
        # ============================================================================
        # DIAGNOSTIC: Track total retrieval time
        # ============================================================================
        print(f"\n⏱️  [read_and_retrieve] START: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        print(f"📝 Query: {user_query[:100]}...")
        total_start = time.time()

        # ============================================================================
        # 🔴 CRITICAL BUG: Invalid parameters!
        # ============================================================================
        # The parameters dense_similarity_top_k, sparse_similarity_top_k, alpha,
        # and enable_reranking are NOT valid for standard LlamaIndex retriever!
        #
        # These parameters are ONLY available when using:
        # 1. Vector stores that support hybrid search (Qdrant, Milvus, Weaviate)
        # 2. Custom retrievers with fusion
        #
        # TODO: Fix this immediately! Replace with valid parameter:
        #       similarity_top_k=5
        # ============================================================================

        # ============================================================================
        # IMPROVEMENT NEEDED: Implement hybrid retrieval (2025 best practice)
        # ============================================================================
        # Research shows +35% accuracy improvement with hybrid search + reranking!
        #
        # OPTION 1 (Quick fix - use now):
        # retriever = index.as_retriever(
        #     similarity_top_k=5,  # Standard parameter that works
        # )
        #
        # OPTION 2 (Advanced - implement later for +35% accuracy):
        # 1. Migrate to Qdrant/Milvus vector store
        # 2. Enable hybrid search (dense + sparse/BM25)
        # 3. Add reranking with Cohere or bge-reranker-large


        # Example advanced retrieval:
        # retriever = index.as_retriever(similarity_top_k=10)
        # nodes = retriever.retrieve(user_query)
        # Apply similarity threshold
        # postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
        # filtered_nodes = postprocessor.postprocess_nodes(nodes)
        # return filtered_nodes[:5]  # Top 5 after filtering
        # ============================================================================

        # ============================================================================
        # IMPROVEMENT NEEDED: Add error handling
        # ============================================================================
        # TODO: Add try-except block
        # TODO: Validate that nodes are returned
        # ============================================================================

        if "EMBEDDING_KEY" not in os.environ:
            load_key()

        # Phase 1: Load index (disk I/O, no API call)
        print("\n📊 Phase 1: Loading index from storage...")
        index = load_embedding_index(path=doc_path)
        

        # Phase 2: Setup retrievers (no API call)
        print("\n📊 Phase 2: Setting up hybrid retriever (Vector + BM25)...")
        setup_start = time.time()
        vector_retriever = index.as_retriever(similarity_top_k=10)
        bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=10)
        fusion_retriever = QueryFusionRetriever([vector_retriever, bm25_retriever],
            similarity_top_k=10
        )
        setup_end = time.time()
        print(f"⏱️  Retriever setup: {(setup_end - setup_start):.2f} seconds")

        # Phase 3: Retrieve nodes (THIS IS WHERE EMBEDDING API CALL HAPPENS!)
        print("\n📊 Phase 3: Retrieving relevant nodes...")
        print("🚨 WARNING: This phase makes an API call to embed the query!")
        print(f"   Endpoint: {os.getenv('OPENAI_API_BASE')}")
        print(f"   Model: text-embedding-3-small")
        retrieve_start = time.time()

        nodes = fusion_retriever.retrieve(user_query)

        retrieve_end = time.time()
        retrieve_duration = retrieve_end - retrieve_start
        print(f"⏱️  Retrieval duration: {retrieve_duration:.2f} seconds")
        print(f"📦 Retrieved {len(nodes)} nodes")

        # Phase 4: Post-processing (local, no API call)
        print("\n📊 Phase 4: Post-processing with similarity cutoff (0.8)...")
        postprocess_start = time.time()
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.8)
        filtered_nodes = postprocessor.postprocess_nodes(nodes)
        postprocess_end = time.time()
        print(f"⏱️  Post-processing: {(postprocess_end - postprocess_start):.2f} seconds")
        print(f"📦 Filtered to {len(filtered_nodes)} nodes")

        # Total time
        total_end = time.time()
        total_duration = total_end - total_start
        print(f"\n⏱️  [read_and_retrieve] TOTAL: {total_duration:.2f} seconds")
        print("="*60)

        return(filtered_nodes)


# ============================================================================
# 🔴 CRITICAL BUG: Debug code executing on module import!
# ============================================================================
# These lines (286-288) execute EVERY TIME this module is imported!
# This causes:
# - Unwanted API calls and costs
# - Slower imports
# - Side effects in production
# - Confusing output when importing the module
#
# TODO: DELETE these lines immediately OR move to:
#       if __name__ == "__main__":
#           # Test code here
#
# PRIORITY: CRITICAL - Fix this first!
# ============================================================================
# nodes = read_and_retrieve("super man")  # 🔴 DELETE THIS LINE
# context_str = "\n\n".join([node.get_content() for node in nodes])  # 🔴 DELETE THIS LINE
# print(context_str)  # 🔴 DELETE THIS LINE
# ============================================================================
# END OF CRITICAL BUG SECTION - DELETE LINES 286-288
# ============================================================================


