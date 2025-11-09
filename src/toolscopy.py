from embedding_loader import retrieve_fitting_instructions, retrieve_products
import time
from datetime import datetime
from langsmith import traceable
from langchain.tools import tool
import logging # Ste 添加logging替代print

# Configure logger for this module
logger = logging.getLogger(__name__)

@tool(
    "retrieve_Fitting_Instructions",
    description=(
        "CALL THIS TOOL FIRST. Retrieves fitting instructions based on user's input information "
        "(swing speed, handicap, skill level, ball flight). Analyzes user metrics to find relevant "
        "club specifications and fitting recommendations. Output guides product selection."
    )
)
@traceable
def retrieve_Fitting_Instructions(query: str) -> str:
    """
    Retrieve fitting instructions based on user input information.

    **CALL THIS TOOL FIRST** - This provides the fitting analysis that guides product selection.

    Analyzes user's physical characteristics and skill metrics to retrieve relevant fitting
    instructions and club specification recommendations from the knowledge base.

    Args:
        query (str): User's metrics and preferences (swing speed, handicap, ball flight, skill level).

    Returns:
        str: Formatted fitting instructions separated by "\n\n--- instructions ---\n\n".

    Example:
        >>> retrieve_Fitting_Instructions("15 handicap, 92 mph swing speed, slice tendency")
        'For 90-95 mph swing: regular flex...\n\n--- instructions ---\n\nSlice correction...'
    """

    logger.info("============================================================")
    logger.info("🎯 TOOL INVOKED: retrieve_Fitting_Instructions")
    logger.info(f"📝 Query: {query}")
    logger.info(f"⏱️  Tool start time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    logger.info("============================================================")

    tool_start = time.time()

    try:
        # ✅ Retrieve fitting instructions as formatted string
        context_str = retrieve_fitting_instructions(query)

        if not context_str:
            logger.warning("⚠️  No results found.")
            return "No relevant fitting instructions found in the knowledge base."

        tool_end = time.time()
        logger.info("============================================================")
        logger.info("✅ TOOL COMPLETED SUCCESSFULLY")
        logger.info(f"📦 Returned formatted fitting instructions context")
        logger.info(f"⏱️  Tool duration: {tool_end - tool_start:.2f} seconds")
        logger.info("============================================================")

        return context_str or "" # Ste 必须返回空字符串

    except Exception as e:
        logger.error("============================================================")
        logger.error("❌ TOOL ERROR")
        logger.error(f"🧠 Exception: {str(e)}", exc_info=True)
        logger.error("============================================================")
        raise Exception(f"retrieve_Fitting_Instructions failed: {str(e)}") # Ste 必须raise异常
    