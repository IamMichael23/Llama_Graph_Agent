import warnings
import os
import logging

# Must suppress warnings BEFORE importing llama_index to catch import-time warnings
# llama_index warning
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
logging.basicConfig(level=logging.ERROR)

# ============================================================================
# IMPROVEMENT NOTE: Add better imports for 2025 best practices
# ============================================================================
# TODO: Add these imports for enhanced retrieval and error handling:
# from tenacity import retry, stop_after_attempt, wait_exponential
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.postprocessor import SimilarityPostprocessor
# from llama_index.postprocessor.cohere_rerank import CohereRerank  # For reranking (+35% accuracy)
# ============================================================================

from abc import ABC, abstractmethod
from typing import Optional, List
from config.load_key import load_key
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.readers.file import PandasCSVReader




def create_md_index(md_files: list, store_path: str, embed_model: OpenAIEmbedding) -> Optional[VectorStoreIndex]:
    """Create vector index from markdown files using MarkdownNodeParser."""
    if not md_files:
        return None

    documents = SimpleDirectoryReader(input_files=md_files).load_data()
    nodes = MarkdownNodeParser().get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)

    path = os.path.join(store_path, "fitting_book_emb")
    os.makedirs(path, exist_ok=True)
    index.storage_context.persist(path)

    print(f"✅ Markdown: {len(documents)} docs → {len(nodes)} chunks → {path}")
    return index


def create_csv_index(csv_files: list, store_path: str, embed_model: OpenAIEmbedding) -> Optional[VectorStoreIndex]:
    """Create vector index from CSV files using PandasCSVReader + SentenceSplitter."""
    if not csv_files:
        return None

    csv_reader = PandasCSVReader(concat_rows = False)
    documents = []
    for csv_file in csv_files:
        documents.extend(csv_reader.load_data(file=csv_file))

    if not documents:
        return None

    nodes = SentenceSplitter(chunk_size=512, chunk_overlap=50).get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)

    path = os.path.join(store_path, "products_emb")
    os.makedirs(path, exist_ok=True)
    index.storage_context.persist(path)

    print(f"✅ CSV: {len(documents)} rows → {len(nodes)} chunks → {path}")
    return index


def create_and_save_embedding_index(load_path: str = "src/raw_data",
                                    store_path: str = "src/storage"):
    """Scan directory, separate by file type, create specialized indexes."""
    if "EMBEDDING_KEY" not in os.environ:
        load_key()

    # Scan and separate files
    md_files, csv_files = [], []
    for root, _, files in os.walk(load_path):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith('.md'):
                md_files.append(path)
            elif file.endswith('.csv'):
                csv_files.append(path)

    print(f"Found: {len(md_files)} markdown, {len(csv_files)} CSV files")

    # Create embedding model
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        api_key=os.getenv("EMBEDDING_KEY"),
        api_base=os.getenv("OPENAI_API_BASE"),
        dimensions=1536
    )

    # Create indexes
    create_md_index(md_files, store_path, embed_model)
    create_csv_index(csv_files, store_path, embed_model)


create_and_save_embedding_index()



