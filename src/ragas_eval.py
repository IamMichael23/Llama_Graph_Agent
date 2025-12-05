"""
Simple RAGAS Evaluation for Golf Equipment Fitting Agent
"""

import os
import csv
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from dotenv import load_dotenv, find_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import retrieve_Fitting_Instructions, retrieve_Fitted_Products
from embedding_loader import retrieve_fitting_instructions, retrieve_products
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_similarity
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from datasets import Dataset
from config.load_key import load_key
from pydantic import SecretStr

load_dotenv(find_dotenv())

# Load API keys (including EMBEDDING_KEY and OPENAI_API_BASE)
load_key()

# Get API keys from environment
api_key = os.getenv("OPENAI_API_KEY")
embedding_key = os.getenv("EMBEDDING_KEY") or api_key  # Fallback to OPENAI_API_KEY if EMBEDDING_KEY not set
api_base = os.getenv("OPENAI_API_BASE")

# Set API keys in environment for RAGAS (it may check os.environ)
os.environ["OPENAI_API_KEY"] = api_key
if embedding_key != api_key:
    os.environ["EMBEDDING_KEY"] = embedding_key
if api_base:
    os.environ["OPENAI_API_BASE"] = api_base

# Create LLM for RAGAS metrics using Langchain ChatOpenAI
# This supports multiple generations required by metrics like answer_relevancy
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    base_url=api_base,
    api_key=SecretStr(api_key) if api_key else None,
)

# Create embedding model using Langchain OpenAIEmbeddings wrapped for RAGAS
# Uses text-embedding-3-large matching embedding_loader.py configuration
langchain_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=embedding_key,
    openai_api_base=api_base
)
embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)


def process_single_question(args):
    """
    Process a single question: retrieve contexts and get agent answer.
    This function runs in a worker process.
    """
    idx, question, system_prompt = args
    
    # Load environment in worker process
    load_dotenv(find_dotenv())
    load_key()
    
    # Get API keys
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    
    try:
        # Create agent for this worker process
        checkpointer = MemorySaver()
        agent = create_react_agent(
            model="openai:gpt-4o-mini",
            tools=[retrieve_Fitting_Instructions, retrieve_Fitted_Products],
            checkpointer=checkpointer,
            prompt=system_prompt
        )
        
        # Get contexts
        fitting_context = retrieve_fitting_instructions(question) or ""
        products_context = retrieve_products(question) or ""
        
        # Prepare context list for RAGAS
        context_list = []
        if fitting_context and fitting_context.strip():
            context_list.append(fitting_context.strip())
        if products_context and products_context.strip():
            context_list.append(products_context.strip())
        if not context_list:
            context_list = [""]
        
        # Run agent
        response = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            {"configurable": {"thread_id": f"eval_{idx}"}}
        )
        answer = response["messages"][-1].content
        
        return {
            'idx': idx,
            'answer': answer,
            'context_list': context_list,
            'success': True
        }
    except Exception as e:
        print(f"Error processing question {idx}: {str(e)}")
        return {
            'idx': idx,
            'answer': f"[ERROR: {str(e)}]",
            'context_list': [""],
            'success': False,
            'error': str(e)
        }


def main():
    # Load system prompt
    prompt_path = os.path.join(os.path.dirname(__file__), "Prompt", "golf_advisor_prompt.md")
    with open(prompt_path, "r") as f:
        system_prompt = f.read()
    
    # Load test cases
    csv_path = os.path.join(os.path.dirname(__file__), "test_dataset.csv")
    questions = []
    ground_truths = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row["question"])
            ground_truths.append(row["ground_truth"])
    
  
    
    print(f"Evaluating {len(questions)} test cases...")
    
    # Check if multiprocessing should be used (can disable with USE_MULTIPROCESSING=false)
    use_multiprocessing = os.getenv("USE_MULTIPROCESSING", "true").lower() == "true"
    
    if use_multiprocessing and len(questions) > 1:
        # Determine number of worker processes
        num_workers = min(cpu_count(), len(questions), int(os.getenv("NUM_WORKERS", cpu_count())))
        print(f"Using {num_workers} parallel workers for faster processing...")
        
        # Prepare arguments for multiprocessing
        process_args = [(i, question, system_prompt) for i, question in enumerate(questions)]
        
        # Process questions in parallel
        print(f"Processing {len(questions)} questions in parallel...")
        start_time = time.time()
        
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_single_question, process_args)
        
        elapsed_time = time.time() - start_time
        print(f"✓ Completed agent processing in {elapsed_time/60:.1f} minutes ({elapsed_time:.1f} seconds)")
        
        # Sort results by index to maintain original order
        results.sort(key=lambda x: x['idx'])
        
        # Extract answers and contexts in correct order
        answers = [r['answer'] for r in results]
        all_contexts = [r['context_list'] for r in results]
        
        # Report any failures
        failures = [r for r in results if not r.get('success', True)]
        if failures:
            print(f"⚠️  {len(failures)} question(s) had errors:")
            for r in failures:
                print(f"   Question {r['idx']+1}: {r.get('error', 'Unknown error')}")
    else:
        # Sequential processing (for debugging or single question)
        print("Processing questions sequentially...")
        checkpointer = MemorySaver()
        agent = create_react_agent(
            model="openai:gpt-4o-mini",
            tools=[retrieve_Fitting_Instructions, retrieve_Fitted_Products],
            checkpointer=checkpointer,
            prompt=system_prompt
        )
        
        answers = []
        all_contexts = []
        
        for i, question in enumerate(questions):
            print(f"Processing {i+1}/{len(questions)}...")
            
            # Get contexts
            fitting_context = retrieve_fitting_instructions(question) or ""
            products_context = retrieve_products(question) or ""
            
            context_list = []
            if fitting_context and fitting_context.strip():
                context_list.append(fitting_context.strip())
            if products_context and products_context.strip():
                context_list.append(products_context.strip())
            if not context_list:
                context_list = [""]
            
            all_contexts.append(context_list)
            
            # Run agent
            response = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                {"configurable": {"thread_id": f"eval_{i}"}}
            )
            answers.append(response["messages"][-1].content)
    
    # Create dataset for RAGAS
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": all_contexts,
        "ground_truth": ground_truths
    })
    
    # Use metrics directly - they should use the LLM configured via ChatOpenAI
    # RAGAS metrics are objects, not callables
    # IMPORTANT: Pass embeddings and llm parameters to avoid default OpenAI model errors
    # ChatOpenAI supports multiple generations required by metrics like answer_relevancy
    print("\nRunning RAGAS evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_similarity],
        llm=llm,
        embeddings=embeddings
    )
    
    # Show results
    df = result.to_pandas()
    metric_columns = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_similarity']
    
    # Filter out rows where ALL metrics are NaN (keep rows with at least one valid metric)
    df_filtered = df.dropna(subset=metric_columns, how='all').copy()
    
    # Calculate means, excluding NaN values from calculations
    print("\nResults (NaN values excluded from calculations):")
    means = df_filtered[metric_columns].mean(skipna=True)
    print(means)
    
    # Count NaN values for each metric
    nan_counts = df_filtered[metric_columns].isna().sum()
    if nan_counts.any():
        print("\nNaN value counts per metric:")
        for col in metric_columns:
            if nan_counts[col] > 0:
                print(f"  {col}: {nan_counts[col]} NaN value(s)")
    
    # Save results (keep all rows, NaN values will be saved as empty or NaN in CSV)
    output_path = os.path.join(os.path.dirname(__file__), "ragas_evaluation_results.csv")
    df_filtered.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    if len(df) != len(df_filtered):
        print(f"  Excluded {len(df) - len(df_filtered)} row(s) where all metrics were NaN")


if __name__ == "__main__":
    main()
