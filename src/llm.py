# src/llm.py
from dotenv import load_dotenv
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain.tools import tool
import json

# 载入环境变量（使用你的 .env）
load_dotenv(".env", override=True)

# ============================================================
# 方案B：最小可运行 + 强顺序 + 调用校验
# 工具都是占位实现，便于你先验证“必须两次调用”的链路
# ============================================================

# ---- 工具1：拟合建议（必须先调用） ----
@tool
def retrieve_Fitting_Instructions(query: str) -> str:
    """
    Analyze user's swing data & tendencies to output fitting guidance.
    MUST be called FIRST. Return a short, deterministic placeholder.
    """
    return (
        "FITTING_INSTRUCTIONS\n"
        f"input: {query}\n"
        "--- instructions ---\n"
        "shaft: X-Stiff\n"
        "loft: 9°\n"
        "length: 45.25\""
    )

# ---- 工具2：匹配产品（在工具1之后调用） ----
@tool
def retrieve_Fitted_Products(query: str) -> str:
    """
    Find concrete products that match the fitting guidance.
    MUST be called AFTER retrieve_Fitting_Instructions.
    """
    return (
        "FITTED_PRODUCTS\n"
        f"criteria: {query}\n"
        "--- product ---\n"
        "TaylorMade Qi10 LS 9° — X-Stiff\n"
        "--- product ---\n"
        "Callaway Paradym Ai-Smoke TD 9° — X-Stiff"
    )

# ---- 构造 Agent：严格提示词，要求两次依序调用 ----
TOOL_USAGE_POLICY = (
    "You are a golf fitting expert.\n"
    "You MUST ALWAYS use BOTH tools in this exact order:\n"
    "  (1) Call retrieve_Fitting_Instructions FIRST with a concise paraphrase of the user's data.\n"
    "  (2) THEN call retrieve_Fitted_Products using the instructions as criteria.\n"
    "Do NOT answer the user until BOTH tool calls have completed.\n"
    "If the user question is vague, still call both tools with best-effort defaults.\n"
)

agent = create_react_agent(
    model="openai:gpt-4o-mini",  # 保持与“教案”一致的字符串模型写法
    tools=[retrieve_Fitting_Instructions, retrieve_Fitted_Products],
    prompt=TOOL_USAGE_POLICY
)

# ============================================================
# 运行与校验
# ============================================================

def _pretty_print_messages(resp):
    print("\n======== FINAL AGENT RESPONSE ========\n")
    print(json.dumps(resp, indent=2, ensure_ascii=False, default=str))
    print("\n======================================\n")

def _assert_both_tools_called(resp):
    """在控制台给出是否两工具均被调用的信号。"""
    msgs = resp.get("messages", [])
    tool_names_seen = []
    for m in msgs:
        # LangChain 的 Tool 消息会带有 name='tool_name'
        # 同时 AI 消息里会有 tool_calls 元数据
        name = getattr(m, "name", None)
        if name in ("retrieve_Fitting_Instructions", "retrieve_Fitted_Products"):
            tool_names_seen.append(name)
    used_1 = "retrieve_Fitting_Instructions" in tool_names_seen
    used_2 = "retrieve_Fitted_Products" in tool_names_seen

    print("---- Tool invocation check ----")
    print(f"retrieve_Fitting_Instructions called? {'✅' if used_1 else '❌'}")
    print(f"retrieve_Fitted_Products called?     {'✅' if used_2 else '❌'}")
    print("--------------------------------\n")

if __name__ == "__main__":
    # 你可以随意替换这一条用户输入
    user_msg = "My driver swing speed is 118 mph, what club should I buy?"

    response = agent.invoke({
        "messages": [
            {"role": "user", "content": user_msg}
        ]
    })

    _pretty_print_messages(response)
    _assert_both_tools_called(response)