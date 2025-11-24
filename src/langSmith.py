import os
import time
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable, Client, evaluate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr
from tools import retrieve_Fitting_Instructions, retrieve_Fitted_Products

load_dotenv(find_dotenv())

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "golf-agent-evaluation"

langsmith_client = Client()

try:
    with open("src/Prompt/golf_advisor_prompt.md", "r") as f:
        system_message = f.read()
except FileNotFoundError:
    system_message = "You are a golf club fitting expert. Provide specific recommendations based on swing data."

llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=SecretStr(os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None,
    timeout=1000
)

agent = create_react_agent(
    model=llm,
    tools=[retrieve_Fitting_Instructions, retrieve_Fitted_Products],
    checkpointer=MemorySaver()
)

dataset_name = "golf-fitting-test-cases"

examples = [
    {"inputs": {"input": "What driver specs for 95 mph swing speed?"}, "outputs": {"expected": "regular flex shaft, 10.5-12 degree loft"}},
    {"inputs": {"input": "High swing speed 115 mph, need low spin driver"}, "outputs": {"expected": "stiff or extra stiff shaft, 8-9 degree loft, low spin head"}},
    {"inputs": {"input": "Senior golfer, 75 mph swing, slice problem"}, "outputs": {"expected": "senior flex shaft, 12+ degree loft, draw bias or offset head"}},
    {"inputs": {"input": "100 mph swing speed, mid handicap player, want more distance"}, "outputs": {"expected": "regular or stiff flex shaft, 9-10.5 degree loft"}},
    {"inputs": {"input": "Professional level, 125 mph driver speed, low ball flight"}, "outputs": {"expected": "extra stiff shaft, 8-9 degree loft, tour level specs"}},
    {"inputs": {"input": "I need a new driver"}, "outputs": {"expected": "need swing speed information, player profile, or fitting data"}},
    {"inputs": {"input": "Best driver for me?"}, "outputs": {"expected": "need more information, swing speed, skill level, or current issues"}},
    {"inputs": {"input": "TaylorMade driver for 105 mph swing speed"}, "outputs": {"expected": "stiff flex shaft, 9-10.5 degree loft, TaylorMade options"}},
    {"inputs": {"input": "Senior flex but 120 mph swing speed"}, "outputs": {"expected": "conflicting information, high swing speed requires stiffer shaft"}},
    {"inputs": {"input": "What shaft weight and flex for 90 mph swing? Also need loft recommendation."}, "outputs": {"expected": "regular flex shaft, 60-70g weight, 10.5-12 degree loft"}}
]

try:
    dataset = langsmith_client.create_dataset(dataset_name, description="Golf club fitting test cases")
    for example in examples:
        langsmith_client.create_example(inputs=example["inputs"], outputs=example["outputs"], dataset_id=dataset.id)
except:
    dataset = langsmith_client.read_dataset(dataset_name=dataset_name)

@traceable(name="golf_fitting_agent_full")
def golf_fitting_agent(input_dict: dict) -> dict:
    query = input_dict["input"]
    thread_id = f"eval_{int(time.time() * 1000)}"
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": thread_id}}
    )
    return {"output": response["messages"][-1].content}

def contains_expected_keywords(run, example) -> dict:
    output = run.outputs["output"].lower()
    keywords = example.outputs["expected"].lower().split()
    matches = sum(1 for keyword in keywords if keyword in output)
    return {"key": "keyword_match", "score": 1 if matches >= len(keywords) * 0.5 else 0, "comment": f"Matched {matches}/{len(keywords)} keywords"}

def has_shaft_and_loft(run, example) -> dict:
    output = run.outputs["output"].lower()
    has_shaft = any(term in output for term in ["flex", "shaft", "regular", "stiff", "senior"])
    has_loft = any(term in output for term in ["loft", "degree", "degrees"])
    return {"key": "has_required_specs", "score": 1 if (has_shaft and has_loft) else 0, "comment": f"Shaft: {has_shaft}, Loft: {has_loft}"}

def handles_edge_cases(run, example) -> dict:
    input_query = example.inputs["input"].lower()
    output = run.outputs["output"].lower()

    is_vague = any(phrase in input_query for phrase in ["i need", "best driver", "help me", "what should"]) and not any(term in input_query for term in ["mph", "swing speed", "handicap"])
    is_conflicting = "senior" in input_query and any(speed in input_query for speed in ["120", "115", "110"])

    if is_vague:
        asks_for_info = any(phrase in output for phrase in ["need more", "information", "swing speed", "tell me", "what is your"])
        return {"key": "edge_case_handling", "score": 1 if asks_for_info else 0, "comment": f"Vague query handled: {asks_for_info}"}
    elif is_conflicting:
        identifies_conflict = any(phrase in output for phrase in ["conflict", "doesn't match", "inconsistent", "typically", "however"])
        return {"key": "edge_case_handling", "score": 1 if identifies_conflict else 0, "comment": f"Conflict identified: {identifies_conflict}"}
    else:
        return {"key": "edge_case_handling", "score": 1, "comment": "Standard query - no special handling needed"}

if __name__ == "__main__":
    results = evaluate(
        golf_fitting_agent,
        data=dataset,
        evaluators=[contains_expected_keywords, has_shaft_and_loft, handles_edge_cases],
        experiment_prefix="golf-agent-eval",
        max_concurrency=1
    )
