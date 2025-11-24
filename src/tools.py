from embedding_loader import retrieve_fitting_instructions, retrieve_products
from langsmith import traceable
from langchain.tools import tool

@tool(
    "retrieve_Products",
    description="CALL AFTER retrieve_Fitting_Instructions. Retrieves golf products matching fitting specs."
)
@traceable
def retrieve_Fitted_Products(query: str) -> str:
    context_str = retrieve_products(query)
    return context_str if context_str else "No products found."


@tool(
    "retrieve_Instructions",
    description="CALL FIRST. Retrieves fitting instructions based on user metrics (swing speed, handicap, ball flight)."
)
@traceable
def retrieve_Fitting_Instructions(query: str) -> str:
    context_str = retrieve_fitting_instructions(query)
    return context_str if context_str else "No fitting instructions found."
