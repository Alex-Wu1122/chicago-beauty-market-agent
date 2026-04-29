import json
import re
import pandas as pd
from openai import OpenAI

# ── Provider config ────────────────────────────────────────────────────────────
_MODEL_CHAIN = [
    {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": "models/gemini-2.5-flash",        "key_name": "GEMINI_API_KEY"},
    {"base_url": "https://api.groq.com/openai/v1",   "model": "llama-3.1-8b-instant",    "key_name": "GROQ_API_KEY"},
    {"base_url": "https://api.cerebras.ai/v1",        "model": "llama3.1-8b",             "key_name": "CEREBRAS_API_KEY"},
    {"base_url": "https://api.groq.com/openai/v1",   "model": "llama-3.3-70b-versatile", "key_name": "GROQ_API_KEY"},
    {"base_url": "https://saniya-briary-xerographically.ngrok-free.dev/v1",  "model": "qwen2.5-7b-instruct-1m",  "key_name": "LOCAL_MODEL_URL"},
]

# ── Tool schemas ───────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "YOU MUST call this in every single response with a non-empty, data-driven analysis. Always cite exact numbers from the dataset. Never omit this call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Your response to the user. Must cite specific numbers from the dataset."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_map_layer",
            "description": "Switch the choropleth map to display a different data layer. Call this in every response based on what the user is asking about.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer": {
                        "type": "string",
                        "enum": ["OpportunityScore", "BeautyPer1000", "RetailPer1000", "MedianIncome", "YoungShare"],
                    }
                },
                "required": ["layer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_zip",
            "description": "Highlight a ZIP code on the map and show its detail card. Call this whenever a specific ZIP is discussed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "zip_code": {"type": "string", "description": "5-digit Chicago ZIP code, e.g. '60622'"}
                },
                "required": ["zip_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "setup_zip_compare",
            "description": "Pre-fill the ZIP Compare tab with two ZIP codes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "zip1": {"type": "string"},
                    "zip2": {"type": "string"},
                },
                "required": ["zip1", "zip2"],
            },
        },
    },
]


def _build_system_prompt(df: pd.DataFrame) -> str:
    top5 = df.nlargest(5, "OpportunityScore")[
        ["ZIP CODE", "OpportunityScore", "MedianIncome", "YoungShare", "RetailPer1000", "BeautyPer1000"]
    ].to_string(index=False)

    data_csv = df[[
        "ZIP CODE", "BeautyCount", "BeautyPer1000", "RetailPer1000",
        "MedianIncome", "YoungShare", "SingleShare", "OpportunityScore",
    ]].to_csv(index=False)

    return f"""You are an AI market analyst embedded in a Chicago Beauty Market Intelligence dashboard.

## Key Correlations (Spearman r vs BeautyPer1000)
- Retail Density: r=0.783 (strongest predictor — beauty businesses cluster with commercial activity)
- Median Income:  r=0.684 (higher income neighborhoods support more beauty spending)
- Young Share:    r=0.594 (25-44 age group are core beauty service consumers)

## Opportunity Score Formula
Weighted composite (0–100): 35% Retail Density + 25% Income + 20% Young Share + 10% Household Size - 25% Competition (BeautyPer1000). Outliers winsorized at p95.

## Top 5 Opportunity ZIPs
{top5}

## Full Dataset (56 Chicago ZIPs)
{data_csv}

## Map Layers
OpportunityScore | BeautyPer1000 | RetailPer1000 | MedianIncome | YoungShare

## CRITICAL INSTRUCTIONS
1. ALWAYS write a substantive text response — explain your reasoning with specific numbers from the data.
2. When asked about a ZIP, give a clear verdict (good/moderate/poor opportunity) with 2-3 specific reasons backed by data.
3. Use tools IN ADDITION to your text — never instead of it.
4. You MUST call these tools in every response:
   • answer — REQUIRED every time. Put your full text response inside answer.text.
   • switch_map_layer — call every response. Pick the most relevant layer:
       - opportunity / best ZIP → OpportunityScore
       - beauty density / competition → BeautyPer1000
       - retail / commercial → RetailPer1000
       - income / wealth → MedianIncome
       - demographics / age → YoungShare
       - general question → OpportunityScore
   • select_zip — call whenever a specific ZIP is mentioned
   • setup_zip_compare — call when comparing two ZIPs
5. ALWAYS cite exact numbers from the dataset in your answer text. Never be vague.
6. FORMAT: Write in a conversational, analytical tone. Use bullet points to highlight key metrics. For comparisons, lead with a verdict sentence, then bullet the key differences, then close with a recommendation.

## Example of a good response (follow this format):
User: "Is 60622 a good place to open a beauty salon?"
Assistant: "Yes — ZIP 60622 (Wicker Park) is the #1 opportunity ZIP with a score of 100.0/100. Here's why:
- **Retail Density: 45.3/1k HH** (top 10% of Chicago) — strong commercial foot traffic
- **Young Adult Share: 42%** — core beauty consumer demographic
- **Median Income: $95,000** — high spending power
- **Competition: 8.2 beauty businesses/1k HH** — moderate, not saturated

Bottom line: high demand, strong foot traffic, affluent young population. A well-positioned salon here would benefit from the existing retail ecosystem."
"""


def _extract_answer_and_actions(tool_calls) -> tuple[str, list]:
    answer = ""
    actions = []
    for tc in tool_calls or []:
        try:
            args = json.loads(tc.function.arguments)
        except Exception:
            args = {}
        name = tc.function.name
        if name == "answer":
            answer = args.get("text", "")
            answer = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), answer)
            answer = answer.replace('\\n', '\n').replace('\\t* ', '\n- ').replace('\\t*', '\n- ').replace('\\t', ' ')
        elif name == "switch_map_layer" and args.get("layer"):
            actions.append({"type": "switch_map_layer", "layer": args["layer"]})
        elif name == "select_zip" and args.get("zip_code"):
            actions.append({"type": "select_zip", "zip_code": args["zip_code"]})
        elif name == "setup_zip_compare" and args.get("zip1") and args.get("zip2"):
            actions.append({"type": "setup_zip_compare", "zip1": args["zip1"], "zip2": args["zip2"]})
    return answer, actions


def chat(messages: list, df: pd.DataFrame, secrets: dict, status_callback=None) -> tuple[str, list]:
    """Call LLM with tool calling. Returns (answer, actions). Falls back through provider chain."""
    system_prompt = _build_system_prompt(df)
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if msg["role"] in ("user", "assistant"):
            api_messages.append({"role": msg["role"], "content": msg["content"]})

    tried = []
    for provider in _MODEL_CHAIN:
        key_name  = provider["key_name"]
        model     = provider["model"]
        api_key   = secrets.get(key_name, "")
        if not api_key:
            tried.append(f"  SKIP  {model} — no API key")
            continue

        base_url = provider["base_url"]
        if key_name == "LOCAL_MODEL_URL":
            api_key = "local"

        if status_callback:
            status_callback(model)
        try:
            client   = OpenAI(api_key=api_key, base_url=base_url)
            extra = {} if key_name == "LOCAL_MODEL_URL" else {"parallel_tool_calls": True}
            response = client.chat.completions.create(
                model=model, messages=api_messages, tools=TOOLS,
                tool_choice="required", max_tokens=1024, timeout=15, **extra,
            )
            if not response.choices:
                print(f"[chatbot] raw response: {response}")
                raise ValueError("Empty choices in response")
            msg_out        = response.choices[0].message
            answer, actions = _extract_answer_and_actions(msg_out.tool_calls)

            if not answer:
                answer = msg_out.content or ""

            if not answer.strip():
                r2 = client.chat.completions.create(
                    model=model,
                    messages=api_messages + [{"role": "user", "content": "Call the 'answer' tool with a non-empty response."}],
                    tools=TOOLS,
                    tool_choice={"type": "function", "function": {"name": "answer"}},
                    max_tokens=1024, timeout=15,
                )
                answer, _ = _extract_answer_and_actions(r2.choices[0].message.tool_calls)
                if not answer:
                    answer = r2.choices[0].message.content or "Sorry, I couldn't generate a response."

            tried.append(f"  OK    {model}")
            print("[chatbot] Provider chain result:")
            for t in tried:
                print(t)
            print(f"         answer : {len(answer)} chars")
            print(f"         actions: {[a['type'] for a in actions]}")
            print(f"         preview: {answer[:120].strip()!r}")
            return answer, actions

        except Exception as e:
            short_err = str(e).split('\n')[0][:120]
            tried.append(f"  FAIL  {model} — {short_err}")
            continue

    print("[chatbot] All providers failed:")
    for t in tried:
        print(t)
    return "Sorry, all AI providers are currently unavailable.", []
