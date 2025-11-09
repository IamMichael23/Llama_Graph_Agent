from dotenv import load_dotenv, find_dotenv
from langgraph.prebuilt.chat_agent_executor import create_react_agent

load_dotenv(".env", override=True)


def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def get_temperature(city: str) -> str:
    """Get temperature for a given city."""
    return f"The temperature in {city} is 20°C."

prompt = prompt = """
You MUST always call BOTH tools:
1. get_weather
2. get_temperature

Call them both once before answering anything.
Never answer directly without using both tools.
"""

agent = create_react_agent(
    model="openai:gpt-4o-mini",  
    tools=[get_weather, get_temperature],  
    prompt=prompt  
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

agent.get_graph().draw_mermaid_png()

print("Full Response: ===============")
import json
print(json.dumps(response, indent=2, ensure_ascii=False, default=str))
print("\n" + "="*50 + "\n")