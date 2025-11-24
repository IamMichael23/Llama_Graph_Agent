"""
Golf Equipment Agent Package

A golf club fitting expert agent with retrieval tools for fitting instructions and products.
"""

from .tools import retrieve_Fitting_Instructions, retrieve_Fitted_Products
from .embedding_loader import retrieve_fitting_instructions, retrieve_products

__version__ = "1.0.0"
__author__ = "Golf Equipment Agent Team"

__all__ = [
    "retrieve_Fitting_Instructions",
    "retrieve_Fitted_Products",
    "retrieve_fitting_instructions",
    "retrieve_products"
]
