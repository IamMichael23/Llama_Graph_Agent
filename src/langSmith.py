import os
import re
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable, Client, evaluate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr
from tools import retrieve_Fitting_Instructions, retrieve_Fitted_Products
import csv


load_dotenv(find_dotenv())

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "golf-agent-evaluation"

client = Client()

# Load prompt
prompt_path = os.path.join(os.path.dirname(__file__), "Prompt", "golf_advisor_prompt.md")
with open(prompt_path, "r") as f:
    system_prompt = f.read()

# Create agent
llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=SecretStr(os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None,
)

agent = create_react_agent(
    model=llm,
    tools=[retrieve_Fitting_Instructions, retrieve_Fitted_Products],
    checkpointer=MemorySaver(),
    prompt=system_prompt
)

# Load test cases
def load_test_cases():
    test_cases = []
    csv_path = os.path.join(os.path.dirname(__file__), "test_dataset.csv")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append({
                "inputs": {"input": row["question"]},
                "outputs": {"answer": row["ground_truth"]}
            })
    return test_cases

# Get or create dataset
dataset_name = "golf-fitting-test-cases"
try:
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Golf equipment fitting test cases"
    )
    examples = load_test_cases()
    client.create_examples(
        inputs=[ex["inputs"] for ex in examples],
        outputs=[ex["outputs"] for ex in examples],
        dataset_id=dataset.id
    )
except:
    dataset = client.read_dataset(dataset_name=dataset_name)

@traceable(name="golf_agent")
def golf_agent(input_dict: dict) -> dict:
    query = input_dict["input"]
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": "eval"}}
    )
    return {"answer": response["messages"][-1].content}

def llm_judge(run, example) -> dict:
    query = example.inputs.get("input", "")
    answer = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")
    
    prompt = f"""You are evaluating a golf equipment fitting recommendation. Compare the actual answer to the expected answer.

**User Question:**
{query}

**Expected Answer (Ground Truth):**
{expected}

**Actual Answer from Agent:**
{answer}

**Simple Evaluation - Check these 3 things:**

1. **Is the flex recommendation correct?** (Swing speed → Flex)
   - Slow (60-85 mph) → Senior/Regular
   - Moderate (85-95 mph) → Regular
   - Fast (95-105 mph) → Stiff
   - Very Fast (105+ mph) → X-stiff/TX

2. **Is the loft recommendation appropriate?** (Swing speed → Loft)
   - Slow speeds need higher loft (12-14°)
   - Fast speeds need lower loft (8-10°)

3. **Is the model type appropriate?** (Handicap → Model)
   - Low handicap (0-9) → Tour/LS
   - Mid handicap (10-18) → Max/Tour-Max
   - High handicap (19+) → Max

**Also check:**
- Does it ask for missing info if the query is vague?
- Does it address conflicts if information contradicts?
- Does it include key specs (flex, loft, model, handedness)?

**Scoring Guide:**
- **9-10**: All key recommendations are correct and complete
- **7-8**: Most recommendations correct, minor issues
- **5-6**: Some correct, but missing important elements
- **3-4**: Major errors in key recommendations
- **0-2**: Fundamentally wrong or completely missing critical info

Respond with ONLY a number from 0-10, followed by why you take score off keep it super concise.

Example:
8.5
Missing loft recommendation for the given swing speed.

"""
    response = llm.invoke(prompt)
    score_text = response.content.strip()
    
    # Extract score (first number found)
    score_match = re.search(r'\d+', score_text)
    score = int(score_match.group()) / 10.0 if score_match else 0.5
    
    return {
        "key": "llm_judge",
        "score": min(max(score, 0.0), 1.0),
        "comment": score_text[:100]
    }

if __name__ == "__main__":
    evaluate(
        golf_agent,
        data="golf-fitting-test-cases",
        evaluators=[llm_judge],
        experiment_prefix="golf-eval-with-prompt",
    )
    print("Done! View at https://smith.langchain.com/")
