# src/llm.py
# ============================================================
# One-file demo: Agent + Tools (with real retrieval if available)
# ============================================================

from dotenv import load_dotenv
load_dotenv(".env", override=True)

import json
import os
import traceback
from typing import Optional

# ⚠️ 保持与你能跑通的版本一致：使用 prebuilt.create_react_agent
# （如果你之后要消除 deprecation warning，可改成 from langchain.agents import create_agent）
from langgraph.prebuilt.chat_agent_executor import create_react_agent

from langchain.tools import tool


# ============================================================
# 🔧 可选：尝试加载你现有的检索函数（若不存在则回退为占位）
# ============================================================

_have_real_retrieval = False
_real_retrieve_fit = None
_real_retrieve_prod = None

try:
    # 你之前贴过的 embedding_loader.py 里应当导出这两个函数
    from embedding_loader import retrieve_fitting_instructions, retrieve_products  # type: ignore
    _real_retrieve_fit = retrieve_fitting_instructions
    _real_retrieve_prod = retrieve_products
    _have_real_retrieval = True
except Exception:
    # 没有就回退
    _have_real_retrieval = False


def _safe_call(func, query: str) -> Optional[str]:
    """安全调用外部检索函数；异常时返回 None。"""
    try:
        if func is None:
            return None
        out = func(query)
        # 你的实现里常直接返回字符串；若返回的是对象，这里统一转字符串
        return out if isinstance(out, str) else str(out)
    except Exception:
        traceback.print_exc()
        return None


# ============================================================
# ✅ 工具 1：先做“配杆建议”检索（或占位）
# ============================================================

@tool
def retrieve_Fitting_Instructions(query: str) -> str:
    """
    Analyze user's swing speed / tendencies and return fitting recommendations.

    MUST BE CALLED FIRST.
    When possible, this tool will call the real retriever from embedding_loader.
    If the real retriever is unavailable, it returns a clearly-marked [FAKE] placeholder.
    """
    # 尝试真实检索
    if _have_real_retrieval:
        text = _safe_call(_real_retrieve_fit, query)
        if text:
            # 明确告诉你这是“真实检索”的结果
            return f"[REAL]\n{text}"
    # 回退占位（可运行、不报错）
    return (
        "[FAKE]\n"
        "Fitting analysis complete.\n"
        f"User query: {query}\n"
        "--- instructions ---\n"
        "Recommended shaft: X-Stiff\n"
        "Recommended loft: 9 degrees\n"
        "Recommended length: 45.25 inches"
    )


# ============================================================
# ✅ 工具 2：根据“配杆建议”去找“具体产品”（或占位）
# ============================================================

@tool
def retrieve_Fitted_Products(query: str) -> str:
    """
    Find concrete golf products that match the fitting recommendations.

    MUST BE CALLED AFTER retrieve_Fitting_Instructions.
    When possible, this tool will call the real product retriever.
    If unavailable, returns a [FAKE] placeholder.
    """
    # 尝试真实检索
    if _have_real_retrieval:
        text = _safe_call(_real_retrieve_prod, query)
        if text:
            return f"[REAL]\n{text}"
    # 回退占位
    return (
        "[FAKE]\n"
        "Matching products found.\n"
        f"Query: {query}\n"
        "--- product ---\n"
        "TaylorMade Qi10 LS Driver — 9° — X-Stiff Shaft\n"
        "--- product ---\n"
        "Callaway Paradym Ai-Smoke Triple Diamond — 9° — X-Stiff"
    )


# ============================================================
# 🧠 构建 Agent（强约束：必须按顺序调用两个工具）
# ============================================================

_SYSTEM_PROMPT = (
    "You are a golf fitting expert.\n"
    "Follow this strict tool-usage policy:\n"
    "1) You MUST call retrieve_Fitting_Instructions FIRST.\n"
    "2) You MUST call retrieve_Fitted_Products SECOND.\n"
    "3) Do NOT answer the user until BOTH tools have been called in that order.\n"
    "4) If a tool returns content starting with [FAKE], explicitly tell the user\n"
    "   that retrieval fell back to a placeholder (indexes may be missing), and\n"
    "   still complete the two-step workflow.\n"
)

agent = create_react_agent(
    model="openai:gpt-4o-mini",                # 和你已跑通的示例保持一致
    tools=[retrieve_Fitting_Instructions, retrieve_Fitted_Products],
    prompt=_SYSTEM_PROMPT
)


# ============================================================
# ▶️ 直接运行（示例）
# ============================================================

if __name__ == "__main__":
    user_msg = "My driver swing speed is 118 mph, what club should I buy?"

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]}
    )

    print("\n======== FINAL AGENT RESPONSE ========\n")
    print(json.dumps(response, indent=2, ensure_ascii=False, default=str))
    print("\n======================================\n")

    # 额外提示：如何判断是否“真检索”
    # - 工具返回文本以 [REAL] 开头 = 真检索成功（走了 embedding_loader）
    # - 以 [FAKE] 开头 = 在本文件里回退的占位输出（没用到你的索引）