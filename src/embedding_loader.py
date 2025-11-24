import os
from config.load_key import load_key
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.postprocessor import SimilarityPostprocessor, LLMRerank
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.llms.openai_like import OpenAILike
from langsmith import traceable

def load_embedding_index(path: str = "src/storage/products_emb/"):
    if "EMBEDDING_KEY" not in os.environ:
        load_key()

    storage_context = StorageContext.from_defaults(persist_dir=path)
    index = load_index_from_storage(
        storage_context,
        embed_model=OpenAIEmbedding(
            model="text-embedding-3-large",
            api_key=os.getenv("EMBEDDING_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            dimensions=1536
        ),
        index_cls=VectorStoreIndex
    )
    return index


@traceable
def retrieve_products(user_query: str):
    return _retrieve_impl(user_query, "src/storage/products_emb/", 0, True)

@traceable
def retrieve_fitting_instructions(user_query: str):
    return _retrieve_impl(user_query, "src/storage/fitting_book_emb/", 0, False)


def _retrieve_impl(user_query: str, doc_path: str, similarity_cutoff: float, use_hybrid: bool):
    if "EMBEDDING_KEY" not in os.environ:
        load_key()

    index = load_embedding_index(path=doc_path)

    if use_hybrid:
        vector_retriever = index.as_retriever(similarity_top_k=15)
        bm25_retriever = index.as_retriever(similarity_top_k=15, retriever_mode="bm25")
        retriever = QueryFusionRetriever([vector_retriever, bm25_retriever], similarity_top_k=10)
    else:
        retriever = index.as_retriever(similarity_top_k=10)

    nodes = retriever.retrieve(user_query)

    postprocessor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
    filtered_nodes = postprocessor.postprocess_nodes(nodes)

    try:
        llm_rerank = OpenAILike(
            model="qwen3-rerank",
            api_base=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
            is_chat_model=False
        )
        reranker = LLMRerank(llm=llm_rerank, top_n=10)
        final_nodes = reranker.postprocess_nodes(filtered_nodes, query_str=user_query)
    except:
        final_nodes = filtered_nodes

    return "\n\n---\n\n".join([node.get_content() for node in final_nodes])


