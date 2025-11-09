from openai import OpenAI

client = OpenAI(
    api_key="xxxx",
    base_url="https://api.agicto.cn/v1"
)

try:
    res = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "test response from tool", "tool_call_id": "123"}
        ]
    )
    print("✅ role=tool ACCEPTED")
    print(res)
except Exception as e:
    print("❌ role=tool REJECTED")
    print(e)