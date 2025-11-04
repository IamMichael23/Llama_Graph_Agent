from langchain_core.tools import tool
from embedding_loader import read_and_query, read_and_retrieve
import time
from datetime import datetime
from langsmith import traceable

# ============================================================================
# IMPROVEMENT NEEDED: Add better imports for error handling
# ============================================================================
# TODO: Add these imports:
# import logging
# from typing import Optional
#
# logger = logging.getLogger(__name__)
# ============================================================================


@tool
@traceable
def query_knowledge_base(query: str) -> str:
    # ============================================================================
    # IMPROVEMENT NEEDED: Enhance docstring (2025 best practice)
    # ============================================================================
    # Research shows: Well-documented tools improve LLM tool selection by 30-40%
    #
    # Current docstring is too vague: "projects, products, or general info"
    # LLM doesn't know when to use this vs other tools
    #
    # TODO: Expand docstring to include:
    # 1. Detailed description of what the tool does
    # 2. Specific use cases (when TO use it)
    # 3. Non-use cases (when NOT to use it)
    # 4. Parameter descriptions with examples
    # 5. Return value description
    # 6. Example usage
    #
    # See IMPLEMENTATION_PLAN.md Section 8 for complete example docstring
    # ============================================================================
    """Use this tool to query our knowledge base about projects, products, or general info."""
    # TODO: Replace with detailed docstring (see IMPLEMENTATION_PLAN.md)

    print("============================================================")
    print("🔍 TOOL INVOKED: query_knowledge_base")
    print(f"📝 Query: {query}")
    print(f"⏱️  Tool start time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    print("============================================================")

    tool_start = time.time()

    # ============================================================================
    # IMPROVEMENT NEEDED: Add error handling
    # ============================================================================
    # TODO: Wrap in try-except block:
    # try:
    #     response = read_and_query(query)
    #     if not response:
    #         return "No information found in knowledge base."
    #     return str(response)
    # except Exception as e:
    #     logger.error(f"Error in query_knowledge_base: {e}")
    #     return f"Error querying knowledge base: {str(e)}"
    # ============================================================================

    response = read_and_query(query)

    tool_end = time.time()
    tool_duration = tool_end - tool_start

    print("============================================================")
    print("✅ TOOL COMPLETED: query_knowledge_base")
    print(f"⏱️  Tool duration: {tool_duration:.2f} seconds")
    print("============================================================")

    return str(response)

# ============================================================================
# 🔴 CRITICAL BUG: Incomplete function!
# ============================================================================
# This function has multiple issues:
# 1. Missing @tool decorator (won't be available to the agent)
# 2. Typo in name: "retrieved" should be "retrieve"
# 3. Missing docstring
# 4. No error handling
# 5. Incorrect return format (should format nodes nicely)
#
# PRIORITY: CRITICAL - Fix before using
# ============================================================================
@tool
@traceable
def retrieve_Fitted_Products(query: str) -> str:  
    """
    Retrieve product recommendations from the knowledge base matching user requirements.

    This tool finds and returns the most relevant products based on user preferences,
    requirements, or characteristics (e.g., swing speed, handicap, playing style, budget).
    It returns raw product information.

    
    """
    # 🔴 MISSING: @tool decorator above this function

    print("============================================================")
    print("🎯 TOOL INVOKED: retrieved_knowledge_base_product")
    print(f"📝 Query: {query}")
    print(f"⏱️  Tool start time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    print("============================================================")

    # ============================================================================
    # IMPROVEMENT NEEDED: Add comprehensive docstring
    # ============================================================================
    # TODO: Add detailed docstring explaining:
    # - This retrieves raw chunks without LLM synthesis (faster than query_knowledge_base)
    # - When to use: Need exact specs, fast retrieval, multiple perspectives
    # - Args and return value descriptions
    # - Examples
    # ============================================================================

    # ============================================================================
    # IMPROVEMENT NEEDED: Add error handling
    # ============================================================================
    # TODO: Wrap in try-except
    # TODO: Check if nodes is empty
    # ============================================================================

    # ============================================================================
    # IMPROVEMENT NEEDED: Format output better
    # ============================================================================
    # TODO: Format nodes with clear separation:
    # if not nodes:
    #     return "No relevant information found."
    # context_str = "\n\n--- Document Chunk ---\n\n".join([node.get_content() for node in nodes])
    # return context_str
    # ============================================================================
    tool_start = time.time()

    try:
        # ✅ This should return a list of nodes or documents
        nodes = read_and_retrieve(query)

        if not nodes:
            print("⚠️  No results found.")
            return "No relevant products found in the knowledge base."

        # ✅ Ensure all contents are strings
        context_str = "\n\n--- Product ---\n\n".join(
            str(node.get_content()) for node in nodes
        )

        tool_end = time.time()
        print("============================================================")
        print("✅ TOOL COMPLETED SUCCESSFULLY")
        print(f"📦 Returned {len(nodes)} product entries")
        for node in nodes:
            print(f"📄 Formatted Content: {node.get_content()}")
        print(f"⏱️  Tool duration: {tool_end - tool_start:.2f} seconds")
        print("============================================================")

        return context_str

    except Exception as e:
        print("============================================================")
        print("❌ TOOL ERROR")
        print(f"🧠 Exception: {str(e)}")
        print("============================================================")
        return f"Error: {str(e)}"
    






@tool
@traceable
def retrieve_Fitting_Instructions(query: str) -> str:
    """
    Retrieve golf club fitting instructions or guidance from the knowledge base.

    This tool searches the indexed fitting and instructional documents to return 
    the most relevant information about club fitting techniques, swing adjustments, 
    or customization processes based on the user's question or situation.

    Use this tool when:
    - The user asks how to perform or interpret a club fitting.
    - The user requests guidance on shaft flex, lie angle, grip size, or launch conditions.
    - The user describes their swing, equipment, or performance metrics and wants fitting advice.
    - The user asks for step-by-step fitting procedures or setup recommendations.

    Do NOT use when:
    - The user requests general golf tips unrelated to fitting (use `query_knowledge_base` instead).
    - The user wants product suggestions or comparisons (use `retrieved_knowledge_base_product` instead).
    - The user asks about store locations or services (use a dedicated service lookup tool if available).

    Args:
        query (str): 
            A natural-language question or request about club fitting, e.g.:
            - "How do I know if I need a stiffer shaft?"
            - "What driver loft should I use for a 105 mph swing speed?"
            - "Explain how to perform a proper club fitting."

    Returns:
        str: 
            Formatted text containing relevant fitting information or procedures 
            retrieved directly from the knowledge base. If no results are found, 
            returns an informative message indicating that no matching data exists.

    Notes:
        - This function performs retrieval-only (no synthesis or summarization).
        - The response reflects raw knowledge base content for accuracy.
        - Used by the LLM agent when precise, instructional, or technical fitting 
          information is needed to support user queries.
    """

    print("============================================================")
    print("🎯 TOOL INVOKED: retrieved_knowledge_base_product")
    print(f"📝 Query: {query}")
    print(f"⏱️  Tool start time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    print("============================================================")

    tool_start = time.time()

    try:
        # ✅ This should return a list of nodes or documents
        nodes = read_and_retrieve(query, doc_path="src/storage/fitting_book_emb/")

        if not nodes:
            print("⚠️  No results found.")
            return "No relevant products found in the knowledge base."

        # ✅ Ensure all contents are strings
        context_str = "\n\n--- Product ---\n\n".join(
            str(node.get_content()) for node in nodes
        )

        tool_end = time.time()
        print("============================================================")
        print("✅ TOOL COMPLETED SUCCESSFULLY")
        print(f"📦 Returned {len(nodes)} product entries")
        for node in nodes:
            print(f"📄 Formatted Content: {node.get_content()}")
        print(f"⏱️  Tool duration: {tool_end - tool_start:.2f} seconds")
        print("============================================================")

        return context_str

    except Exception as e:
        print("============================================================")
        print("❌ TOOL ERROR")
        print(f"🧠 Exception: {str(e)}")
        print("============================================================")
        return f"Error: {str(e)}"
    




# CORRECTED VERSION (TODO: Replace above function with this)
# ============================================================================
# @tool
# def retrieve_knowledge_base(query: str) -> str:
#     """
#     Retrieve raw document chunks from the golf equipment knowledge base.
#
#     This tool retrieves relevant text chunks directly from documents without
#     LLM synthesis. It's faster than query_knowledge_base and returns exact
#     text from source documents.
#
#     Use this tool when:
#     - You need exact specifications or quotes from documents
#     - You want multiple perspectives on a topic
#     - Speed is important (no LLM synthesis overhead)
#     - You want to see all relevant information before synthesizing
#
#     Do NOT use when:
#     - User wants a direct answer (use query_knowledge_base instead)
#     - You need the LLM to interpret or summarize information
#
#     Args:
#         query: Search query to find relevant document chunks.
#                Examples: "driver loft specifications", "TaylorMade Qi35",
#                         "shaft flex for high swing speed"
#
#     Returns:
#         Formatted text chunks from the most relevant documents, separated
#         by "--- Document Chunk ---". Returns error message if retrieval fails.
#
#     Example:
#         >>> retrieve_knowledge_base("TaylorMade Qi35 specifications")
#         "--- Document Chunk ---\n\nThe TaylorMade Qi35 driver features..."
#     """
#     try:
#         nodes = read_and_retrieve(query)
#         if not nodes:
#             return "No relevant information found in the knowledge base for this query."
#
#         # Format nodes with clear separation
#         context_str = "\n\n--- Document Chunk ---\n\n".join([
#             node.get_content() for node in nodes
#         ])
#         return context_str
#     except Exception as e:
#         return f"Error retrieving from knowledge base: {str(e)}"
# ============================================================================


