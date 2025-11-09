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
    # ✅ FIXED: Model consistency
    # ============================================================================
    # Both create and load functions now use: text-embedding-3-large
    #
    # Research (2025): text-embedding-3-large achieves 80.5% accuracy
    #                  text-embedding-3-small achieves 75.8% accuracy
    #                  Cost: $0.13/M vs $0.02/M tokens (worth it for accuracy)
    # ============================================================================

    print("📂 Loading index from disk...")
    storage_context = StorageContext.from_defaults(persist_dir=path)
    index = load_index_from_storage(storage_context,
                                    embed_model=OpenAIEmbedding(
                                        model = "text-embedding-3-large",
                                        api_key=os.getenv("EMBEDDING_KEY"),
                                        api_base=os.getenv("OPENAI_API_BASE"),
                                        dimensions=1536),
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
def retrieve_products(user_query: str = "what do we have?"):
        """
        Retrieve product information using hybrid BM25 + Vector retrieval.
        Optimized for structured product data with exact specifications.

        Args:
            user_query: Query about golf products/equipment

        Returns:
            list: Retrieved and filtered product nodes
        """
        return _retrieve_impl(user_query, "src/storage/products_emb/", similarity_cutoff=0, use_hybrid=True, func_name="retrieve_products")


@traceable
def retrieve_fitting_instructions(user_query: str = "what do we have?"):
        """
        Retrieve fitting instructions using pure vector semantic search.
        Optimized for instructional/conceptual content.

        Args:
            user_query: Query about fitting guidance/instructions

        Returns:
            list: Retrieved and filtered instruction nodes
        """
        return _retrieve_impl(user_query, "src/storage/fitting_book_emb/", similarity_cutoff=0, use_hybrid=False, func_name="retrieve_fitting_instructions")


def _retrieve_impl(user_query: str, doc_path: str, similarity_cutoff: float, use_hybrid: bool, func_name: str = "_retrieve_impl"):
        # ============================================================================
        # DIAGNOSTIC: Track total retrieval time
        # ============================================================================
        print(f"\n⏱️  [{func_name}] START: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
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
        if use_hybrid:
            print("\n📊 Phase 2: Setting up HYBRID retriever (Vector + BM25)...")
            setup_start = time.time()
            vector_retriever = index.as_retriever(similarity_top_k=5)
            bm25_retriever = index.as_retriever(similarity_top_k=5, retriever_mode="bm25")
            fusion_retriever = QueryFusionRetriever([vector_retriever, bm25_retriever],
                similarity_top_k=3
            )
            retriever = fusion_retriever
            setup_end = time.time()
            print(f"⏱️  Retriever setup: {(setup_end - setup_start):.2f} seconds")
        else:
            print("\n📊 Phase 2: Setting up VECTOR-ONLY retriever (Semantic Search)...")
            setup_start = time.time()
            retriever = index.as_retriever(similarity_top_k=5)
            setup_end = time.time()
            print(f"⏱️  Retriever setup: {(setup_end - setup_start):.2f} seconds")

        # Phase 3: Retrieve nodes (THIS IS WHERE EMBEDDING API CALL HAPPENS!)
        print("\n📊 Phase 3: Retrieving relevant nodes...")
        print("🚨 WARNING: This phase makes an API call to embed the query!")
        print(f"   Endpoint: {os.getenv('OPENAI_API_BASE')}")
        print(f"   Model: text-embedding-3-large")
        retrieve_start = time.time()

        nodes = retriever.retrieve(user_query)

        retrieve_end = time.time()
        retrieve_duration = retrieve_end - retrieve_start
        print(f"⏱️  Retrieval duration: {retrieve_duration:.2f} seconds")
        print(f"📦 Retrieved {len(nodes)} nodes")

        # Phase 4: Post-processing (local, no API call)
        print(f"\n📊 Phase 4: Post-processing with similarity cutoff ({similarity_cutoff})...")
        postprocess_start = time.time()
        postprocessor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        filtered_nodes = postprocessor.postprocess_nodes(nodes)
        postprocess_end = time.time()
        print(f"⏱️  Post-processing: {(postprocess_end - postprocess_start):.2f} seconds")
        print(f"📦 Filtered to {len(filtered_nodes)} nodes")

        # Total time
        total_end = time.time()
        total_duration = total_end - total_start
        print(f"\n⏱️  [{func_name}] TOTAL: {total_duration:.2f} seconds")
        print("="*60)

        # Convert filtered_nodes to formatted string
        context_str = "\n\n---\n\n".join([node.get_content() for node in filtered_nodes])
        return context_str


# Backward compatibility - deprecated, use retrieve_products() or retrieve_fitting_instructions()
@traceable
def read_and_retrieve(user_query: str = "what do we have?", doc_path: str = "src/storage/products_emb/"):
    """
    DEPRECATED: Use retrieve_products() or retrieve_fitting_instructions() instead.
    This function is kept for backward compatibility only.
    """
    print("⚠️  WARNING: read_and_retrieve() is deprecated. Use retrieve_products() or retrieve_fitting_instructions() instead.")

    # Determine which function to use based on doc_path
    if "products" in doc_path:
        return retrieve_products(user_query)
    elif "fitting" in doc_path:
        return retrieve_fitting_instructions(user_query)
    else:
        # Default to products
        return retrieve_products(user_query)


# ============================================================================
# For testing, uncomment the block below and run with: python -m src.embedding_loader
# ============================================================================
if __name__ == "__main__":
    # print("\n" + "="*60)
    # print("TESTING PRODUCT RETRIEVAL (Hybrid BM25 + Vector)")
    # print("="*60)
    # product_nodes = retrieve_products("Driver with 9° loft for right-hand players")
    # product_context = "\n\n".join([node.get_content() for node in product_nodes])
    # print(product_context)

    print("\n" + "="*60)
    print("TESTING FITTING INSTRUCTIONS RETRIEVAL (Vector Only)")
    print("="*60)
    instruction_nodes = retrieve_fitting_instructions("How to fit a driver for high swing speed")
    
    print(instruction_nodes)
# ============================================================================


