"""Tool-calling smoke test against the served Qwen3.8-Flash-Next package (OpenAI-compatible API)."""
import json, os, sys, time, urllib.request

BASE = os.environ.get("QWEN38_ENDPOINT", "http://127.0.0.1:20000"); MODEL = "Qwen/Qwen3.8-Flash-Next"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {
    "name": "get_time", "description": "Get the current local time in a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

def post(body, stream=False):
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        if not stream:
            return json.load(r), time.time() - t0
        chunks = []
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
        return chunks, time.time() - t0

results = {}
# 1. plain tool call, non-streaming
body = {"model": MODEL, "messages": [{"role": "user", "content": "What is the weather in Toronto right now? Use celsius."}],
        "tools": TOOLS, "tool_choice": "auto", "temperature": 0.0, "max_tokens": 512}
resp, dt = post(body)
msg = resp["choices"][0]["message"]
print("== 1. non-streaming tool call  (%.1fs)" % dt)
print("finish_reason:", resp["choices"][0]["finish_reason"])
print("content:", repr((msg.get("content") or "")[:200]))
print("reasoning_content:", repr((msg.get("reasoning_content") or msg.get("reasoning") or "")[:160]))
print("tool_calls:", json.dumps(msg.get("tool_calls"), indent=None)[:600])
tc = (msg.get("tool_calls") or [None])[0]
results["single_call"] = bool(tc and tc["function"]["name"] == "get_weather" and "Toronto" in tc["function"]["arguments"])
ok_json = False
try:
    args = json.loads(tc["function"]["arguments"]) if tc else None; ok_json = isinstance(args, dict)
except Exception as e:  # noqa
    print("arguments not JSON:", e)
results["arguments_json"] = ok_json

# 2. tool result round trip -> final answer
if tc:
    msgs = body["messages"] + [msg, {"role": "tool", "tool_call_id": tc["id"], "name": "get_weather",
                                      "content": json.dumps({"city": "Toronto", "temperature_c": 17, "condition": "light rain"})}]
    resp2, dt2 = post({"model": MODEL, "messages": msgs, "tools": TOOLS, "temperature": 0.0, "max_tokens": 512})
    m2 = resp2["choices"][0]["message"]
    print("\n== 2. after tool result  (%.1fs)" % dt2)
    print("finish_reason:", resp2["choices"][0]["finish_reason"], "| tool_calls:", m2.get("tool_calls"))
    print("content:", repr((m2.get("content") or "")[:300]))
    c = (m2.get("content") or "")
    results["final_answer_uses_result"] = ("17" in c and "rain" in c.lower()) and not m2.get("tool_calls")

# 3. two calls in one turn
resp3, dt3 = post({"model": MODEL, "messages": [{"role": "user", "content": "Tell me both the weather and the current time in Paris."}],
                   "tools": TOOLS, "tool_choice": "auto", "temperature": 0.0, "max_tokens": 512})
m3 = resp3["choices"][0]["message"]; names = [t["function"]["name"] for t in (m3.get("tool_calls") or [])]
print("\n== 3. parallel tool calls  (%.1fs)" % dt3, "->", names, "content:", repr((m3.get("content") or "")[:100]))
results["parallel_calls"] = sorted(names) == ["get_time", "get_weather"]

# 4. streaming tool call
chunks, dt4 = post({"model": MODEL, "messages": [{"role": "user", "content": "What time is it in Tokyo?"}],
                    "tools": TOOLS, "tool_choice": "auto", "temperature": 0.0, "max_tokens": 512, "stream": True}, stream=True)
name = ""; args = ""; fr = None; reasoning = 0; content = ""
for ch in chunks:
    for c in ch.get("choices", []):
        d = c.get("delta", {}); fr = c.get("finish_reason") or fr
        reasoning += len(d.get("reasoning_content") or d.get("reasoning") or ""); content += d.get("content") or ""
        for t in d.get("tool_calls") or []:
            f = t.get("function", {}); name += f.get("name") or ""; args += f.get("arguments") or ""
print("\n== 4. streaming  (%.1fs, %d chunks)" % (dt4, len(chunks)), "finish:", fr, "name:", name, "args:", args, "reasoning chars:", reasoning, "content:", repr(content[:80]))
results["streaming_call"] = name == "get_time" and "Tokyo" in args and fr == "tool_calls"

# 5. no tools needed -> plain answer (no spurious tool call)
resp5, dt5 = post({"model": MODEL, "messages": [{"role": "user", "content": "What is 7 multiplied by 8? Answer in one sentence."}],
                   "tools": TOOLS, "tool_choice": "auto", "temperature": 0.0, "max_tokens": 256})
m5 = resp5["choices"][0]["message"]
print("\n== 5. no tool needed  (%.1fs)" % dt5, "tool_calls:", m5.get("tool_calls"), "content:", repr((m5.get("content") or "")[:120]))
results["no_spurious_call"] = not m5.get("tool_calls") and "56" in (m5.get("content") or "")

print("\nRESULTS", json.dumps(results))
print("ALL PASS" if all(results.values()) else "SOME FAILED")
