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
import pandas as pd
from llama_index.core.schema import Document


def format_golf_club_row(row_dict: dict) -> tuple[str, dict]:
    """
    Convert a CSV row into natural language text with structured metadata.

    Args:
        row_dict: Dictionary representing one CSV row with columns:
                  club, number, loft, hand, lie, volume, length, sw, Type_Check, Type

    Returns:
        tuple: (formatted_text, metadata_dict)
    """
    # Type mapping to fix incorrect/missing types
    type_mapping = {
        'Driver': 'Driver',
        'LS Driver': 'Driver',
        'Max Driver': 'Driver',
        'Lite Driver': 'Driver',
        'Fairway': 'Fairway',
        'Tour Fairway': 'Fairway',
        'Rescue': 'Hybrid',
        'Iron': 'Iron',
        'HL Iron': 'Iron',
        'P·790': 'Iron',
        'P·770': 'Iron',
        'P·7CB': 'Iron',
        'P·7MB': 'Iron',
        'P·7TW': 'Iron',
        'MG5 LB': 'Wedge',
        'MG5 SC': 'Wedge',
        'MG5 SB': 'Wedge',
        'MG5 SX': 'Wedge',
        'MG5 HB': 'Wedge',
        'MG5 TW': 'Wedge',
        'ATS': 'Wedge',
        'ATV': 'Wedge',
        'ATX': 'Wedge',
        'ATC': 'Wedge',
        'ATW': 'Wedge',
        'P·UDI': 'Hybrid',
        'P·DHY': 'Hybrid',
    }

    # Extract values, handle NaN
    club = str(row_dict.get('club', '')).strip()
    number = str(row_dict.get('number', '')).strip() if pd.notna(row_dict.get('number')) else ''
    loft = str(row_dict.get('loft', '')).strip()
    hand = str(row_dict.get('hand', '')).strip()
    lie = str(row_dict.get('lie', '')).strip()
    volume = str(row_dict.get('volume', '')).strip() if pd.notna(row_dict.get('volume')) else ''
    length = str(row_dict.get('length', '')).strip()
    sw = str(row_dict.get('sw', '')).strip()
    product_type = str(row_dict.get('Type', 'Unknown')).strip()

    # Apply intelligent type mapping based on club name
    # Extract base club name (before any spaces or numbers)
    club_base = club.split()[0] if club else ''

    # Try exact match first, then partial match
    if club in type_mapping:
        product_type = type_mapping[club]
    else:
        # Try matching against club base name or partial names
        for key, mapped_type in type_mapping.items():
            if key in club:
                product_type = mapped_type
                break

    # Build natural language description
    parts = []

    # Main club name and number
    if club and number:
        parts.append(f"{club} number {number}")
    elif club:
        parts.append(club)

    # Loft
    if loft:
        parts.append(f"with {loft} loft")

    # Hand orientation - normalize to RH/LH and expand for readability
    if hand:
        # Normalize hand notation (handle variations like "right hand", "Right", etc.)
        hand_normalized = hand.strip().upper()

        # Map variations to standard RH/LH
        hand_map = {
            'RIGHT': 'RH',
            'RIGHT HAND': 'RH',
            'RIGHT-HAND': 'RH',
            'R': 'RH',
            'LEFT': 'LH',
            'LEFT HAND': 'LH',
            'LEFT-HAND': 'LH',
            'L': 'LH'
        }

        # Normalize if needed
        if hand_normalized in hand_map:
            hand_normalized = hand_map[hand_normalized]

        # Handle combination (RH/LH or LH/RH)
        if '/' in hand_normalized:
            # Ensure consistent order: RH/LH
            hand_parts = [h.strip() for h in hand_normalized.split('/')]
            hand_parts = [hand_map.get(h, h) for h in hand_parts]  # Normalize each part

            if 'RH' in hand_parts and 'LH' in hand_parts:
                hand_normalized = 'RH/LH'  # Consistent order
                parts.append(f"available for right-hand and left-hand players")
            elif 'RH' in hand_parts:
                hand_normalized = 'RH'
                parts.append(f"for right-hand players")
            elif 'LH' in hand_parts:
                hand_normalized = 'LH'
                parts.append(f"for left-hand players")
        elif hand_normalized == 'RH':
            parts.append(f"for right-hand players")
        elif hand_normalized == 'LH':
            parts.append(f"for left-hand players")

        # Store normalized value in metadata
        hand = hand_normalized

    # Technical specs
    tech_specs = []
    if lie:
        tech_specs.append(f"lie angle {lie}")
    if volume:
        tech_specs.append(f"clubhead volume {volume}")
    if length:
        tech_specs.append(f"length {length}")
    if sw:
        tech_specs.append(f"swing weight {sw}")

    if tech_specs:
        parts.append(f"with specifications: {', '.join(tech_specs)}")

    # Product type
    parts.append(f"Product category: {product_type}")

    formatted_text = " ".join(parts) + "."

    # Create metadata for filtering
    # Note: LlamaIndex metadata should be strings or JSON-serializable types
    metadata = {
        'club_name': club,
        'product_type': product_type,
        'hand': hand,
        'loft_str': loft,  # Keep original loft string
    }

    # Add numeric loft if parseable (stored as string for metadata compatibility)
    if loft:
        # Extract numeric part from loft (e.g., "9°" -> 9)
        try:
            loft_num = float(loft.replace('°', '').strip())
            metadata['loft_numeric'] = str(loft_num)  # Store as string for metadata
        except ValueError:
            # Warn but don't fail - loft might be a range like "56°–60°"
            print(f"⚠️  Warning: Could not parse loft '{loft}' as single number for club '{club}'")

    if number:
        metadata['club_number'] = number

    return formatted_text, metadata


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
    """Create vector index from CSV files with natural language formatting and metadata."""
    if not csv_files:
        return None

    # Read CSV files into DataFrames and format each row
    formatted_documents = []
    total_rows = 0

    for csv_file in csv_files:
        print(f"📄 Processing CSV: {csv_file}")
        df = pd.read_csv(csv_file)

        # Validate required columns exist
        required_columns = {'club', 'loft', 'hand', 'Type'}
        actual_columns = set(df.columns)
        missing_columns = required_columns - actual_columns

        if missing_columns:
            raise ValueError(
                f"❌ CSV file '{csv_file}' is missing required columns: {missing_columns}\n"
                f"   Found columns: {list(df.columns)}\n"
                f"   Required: {required_columns}"
            )

        print(f"✅ CSV validation passed - all required columns present")

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            formatted_text, metadata = format_golf_club_row(row_dict)

            # Create Document with formatted text and metadata
            doc = Document(
                text=formatted_text,
                metadata=metadata,
                id_=f"product_{total_rows}"
            )
            formatted_documents.append(doc)
            total_rows += 1

            # Debug: Print first 3 formatted documents
            if total_rows <= 3:
                print(f"\n🔍 Example {total_rows}:")
                print(f"   Original: {row_dict}")
                print(f"   Formatted: {formatted_text}")
                print(f"   Metadata: {metadata}")

    if not formatted_documents:
        return None

    print(f"\n✅ Formatted {len(formatted_documents)} products into natural language")

    # Create nodes from formatted documents
    # Use larger chunks since we already have formatted text
    nodes = SentenceSplitter(chunk_size=1024, chunk_overlap=100).get_nodes_from_documents(formatted_documents)

    # Create index
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)

    path = os.path.join(store_path, "products_emb")
    os.makedirs(path, exist_ok=True)
    index.storage_context.persist(path)

    print(f"✅ CSV: {len(formatted_documents)} products → {len(nodes)} chunks → {path}")
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
    # create_md_index(md_files, store_path, embed_model)  # Commented out - only creating CSV index
    create_csv_index(csv_files, store_path, embed_model)


def embed_golf_csv_only():
    """
    Embed only the Golf_equ_list_typed.csv file.
    Quick function to create product embeddings without scanning for other files.
    """
    if "EMBEDDING_KEY" not in os.environ:
        load_key()

    # Specific CSV file path
    csv_file = "src/raw_data/Golf_equ_list_typed.csv"

    print(f"\n{'='*60}")
    print(f"🎯 Embedding Single CSV File")
    print(f"{'='*60}")
    print(f"📄 File: {csv_file}\n")

    # Check if file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"❌ CSV file not found: {csv_file}")

    # Create embedding model
    print("🔧 Creating embedding model...")
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        api_key=os.getenv("EMBEDDING_KEY"),
        api_base=os.getenv("OPENAI_API_BASE"),
        dimensions=1536
    )
    print("✅ Embedding model ready\n")

    # Create CSV index
    create_csv_index([csv_file], "src/storage", embed_model)

    print(f"\n{'='*60}")
    print("🎉 CSV Embedding Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Run the CSV-only embedding function
    embed_golf_csv_only()



