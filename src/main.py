import os
import time
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import openai
from tools import retrieve_Fitting_Instructions, retrieve_Fitted_Products

load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found")

# Load the system prompt

prompt_path = os.path.join(os.path.dirname(__file__), "Prompt", "golf_advisor_prompt.md")
with open(prompt_path, "r") as f:
        system_prompt = f.read()
checkpointer = MemorySaver()

agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[retrieve_Fitting_Instructions, retrieve_Fitted_Products],
    checkpointer=checkpointer,
    prompt=system_prompt
)
def run_agent(user_query: str, thread_id: str = None):
    if not thread_id:
        thread_id = f"session_{int(time.time())}"

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]},
        {"configurable": {"thread_id": thread_id}}
    )

    print(response["messages"][-1].content)
    return response

def chat():
    thread_id = f"session_{int(time.time())}"
    print("\nGolf Equipment Agent")
    print("Type 'quit' or 'exit' to end\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if user_input:
                run_agent(user_input, thread_id)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    chat()
