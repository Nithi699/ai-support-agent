"""
Customer Support AI Agent
==========================================
Run locally (after filling in config values):
    uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
    agentcore deploy

Invoke deployed agent:
    agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
@tool
def get_page_title(url: str) -> str:
    """Open a web page in a remote browser and return its title.

    Args:
        url: Full URL to open, for example https://www.amazon.com
    """
    import json as _json
    import time as _time
    from bedrock_agentcore.tools.browser_client import BrowserClient
    from websockets.sync.client import connect

    client = BrowserClient(REGION)
    try:
        client.start()
        ws_url, headers = client.generate_ws_headers()
        with connect(ws_url, additional_headers=headers, open_timeout=20, max_size=None) as ws:
            counter = [0]

            def send(method, params=None, session_id=None):
                counter[0] += 1
                msg = {"id": counter[0], "method": method, "params": params or {}}
                if session_id:
                    msg["sessionId"] = session_id
                ws.send(_json.dumps(msg))
                while True:
                    data = _json.loads(ws.recv(timeout=30))
                    if data.get("id") == counter[0]:
                        if "error" in data:
                            raise RuntimeError(str(data["error"]))
                        return data.get("result", {})

            target = send("Target.createTarget", {"url": url})["targetId"]
            sid = send("Target.attachToTarget", {"targetId": target, "flatten": True})["sessionId"]
            title = ""
            for _ in range(20):
                res = send("Runtime.evaluate", {"expression": "document.readyState + '|' + document.title"}, sid)
                value = res.get("result", {}).get("value", "")
                state, _sep, title = value.partition("|")
                if state == "complete" and title:
                    break
                _time.sleep(1)
            return f"Page title: {title}" if title else "Page loaded but no title was found."
    except Exception as e:
        return f"Browser error: {e}"
    finally:
        try:
            client.stop()
        except Exception:
            pass



logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
# This starter uses an unsigned MCP connection and therefore assumes the
# project Gateway is configured with the NONE authorizer.
GATEWAY_URL = "https://customersupportgateway-ymqfdhpmyp.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "DRPK8NNCOB"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-OUE5Ak98C3"


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
# Returns a dict mapping strategy type to namespace template string, e.g.
#   { "SEMANTIC": "cs_agent/{actorId}/facts",
#     "USER_PREFERENCE": "cs_agent/{actorId}/preferences" }

def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    strategies_resp = mem_client.get_memory_strategies(memory_id)
    strategies = strategies_resp if isinstance(strategies_resp, list) else strategies_resp.get("memoryStrategies", [])
    result = {}
    for strat in strategies:
        strat_type = strat.get("memoryStrategyType")
        templates = strat.get("namespaceTemplates")
        if templates:
            ns = templates[0]
        else:
            namespaces = strat.get("namespaces", [])
            ns = namespaces[0] if namespaces else ""
        if strat_type and ns:
            result[strat_type] = ns
    return result


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        messages = event.agent.messages
        if not messages:
            return

        last_msg = messages[-1]
        # Only plain-text user messages (not tool results or assistant messages)
        if last_msg.get("role") != "user":
            return

        content = last_msg.get("content")
        if not isinstance(content, list) or not content:
            return

        text_block = content[0]
        if not isinstance(text_block, dict) or "text" not in text_block:
            return

        query = text_block["text"]
        # Skip if context was already injected
        if "Customer Context:" in query:
            return

        collected_memories = []
        for strat_type, ns_template in self.namespaces.items():
            namespace = ns_template.replace("{actorId}", self.actor_id)
            try:
                mems = self.memory_client.retrieve_memories(
                    self.memory_id, namespace, query, top_k=5
                )
                # retrieve_memories returns a list of records (fall back to dict form)
                records = mems if isinstance(mems, list) else mems.get("memorySummaries", [])
                for m in records:
                    mem_content = m.get("content", {})
                    mem_text = (
                        mem_content.get("text", "") if isinstance(mem_content, dict) else str(mem_content)
                    ).strip()
                    if mem_text:
                        collected_memories.append(f"[{strat_type}] {mem_text}")
            except Exception as e:
                logger.warning(f"Failed to retrieve memory for {namespace}: {e}")

        if collected_memories:
            memories_str = "\n".join(collected_memories)
            text_block["text"] = f"Customer Context:\n{memories_str}\n\n{query}"

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        messages = event.agent.messages

        user_query = None
        assistant_response = None

        # Walk backwards to find the last assistant response and last user query
        for msg in reversed(messages):
            role = msg.get("role")
            content = msg.get("content")
            if not isinstance(content, list) or not content:
                continue
            text_val = content[0].get("text") if isinstance(content[0], dict) else None

            if role == "assistant" and not assistant_response and text_val:
                assistant_response = text_val
            elif role == "user" and not user_query and text_val:
                # Remove the injected context header for clean storage
                if "Customer Context:" in text_val:
                    parts = text_val.split("\n\n")
                    user_query = parts[-1] if len(parts) > 1 else text_val
                else:
                    user_query = text_val

            if user_query and assistant_response:
                break

        if user_query and assistant_response:
            try:
                self.memory_client.create_event(
                    self.memory_id,
                    self.actor_id,
                    self.session_id,
                    messages=[(user_query, "USER"), (assistant_response, "ASSISTANT")],
                )
            except Exception as e:
                logger.warning(f"Failed to save support interaction to memory: {e}")

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID:
        return "Knowledge base not configured."

    try:
        response = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
        results = response.get("retrievalResults", [])
        if not results:
            return "No relevant information found in the knowledge base."

        chunks = [res.get("content", {}).get("text", "") for res in results]
        return "\n---\n".join([c for c in chunks if c])
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = f"""
import json

loyalty_points = {loyalty_points}
tier = "{tier}"
order_total = {order_total}
product_category = "{product_category}"

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

# 1 point = $0.01 value, max redemption is 50% of order total, rounded down to a multiple of 500 points
max_redeemable_points = int((order_total * 0.50) / 0.01)
usable_points = min(loyalty_points, max_redeemable_points)
points_redeemed = (usable_points // 500) * 500
points_discount = points_redeemed * 0.01

# Tier discount is applied to the subtotal after points
subtotal_after_points = max(0.0, order_total - points_discount)
tier_rate = tier_rates.get(tier, 0.00)
tier_discount = subtotal_after_points * tier_rate

final_total = max(0.0, subtotal_after_points - tier_discount)
total_savings = order_total - final_total

earn_rate = earn_rates.get(product_category, 1)
points_earned = int(final_total * earn_rate)
# remaining_points = balance minus points redeemed (points earned are reported separately)
remaining_points = loyalty_points - points_redeemed

result = {{
    "order_total": round(order_total, 2),
    "points_redeemed": points_redeemed,
    "points_discount": round(points_discount, 2),
    "tier": tier,
    "tier_discount": round(tier_discount, 2),
    "final_total": round(final_total, 2),
    "total_savings": round(total_savings, 2),
    "points_earned": points_earned,
    "remaining_points": remaining_points
}}
print(json.dumps(result))
"""

    try:
        with code_session(REGION) as code_client:
            response = code_client.invoke(
                "executeCode",
                {"language": "python", "code": code, "clearContext": True},
            )
            for event in response["stream"]:
                return json.dumps(event["result"])
        return json.dumps({"error": "No output returned from code interpreter session."})

    except Exception as e:
        # Fallback: tier discount only, if the Code Interpreter is unavailable
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        tier_rate = tier_rates.get(tier, 0.00)
        tier_discount = order_total * tier_rate
        final_total = max(0.0, order_total - tier_discount)
        return json.dumps({
            "order_total": round(order_total, 2),
            "tier": tier,
            "tier_discount": round(tier_discount, 2),
            "final_total": round(final_total, 2),
            "fallback_used": True,
            "error": str(e),
        })


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert, empathetic, and efficient Customer Support AI Agent. "
    "You help customers with product catalog details, return policies, warranty queries, "
    "order statuses, loyalty points, and tier discounts. Use the provided tools when necessary "
    "to look up accurate information or perform calculations. Always maintain a professional tone."
)


@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        # 1. Extract payload fields
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id", "ANONYMOUS")
        session_id = payload.get("session_id") or str(uuid.uuid4())

        # 2. Memory hook for this actor/session
        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        # 3. Browser tool
        # browser handled by get_page_title tool
        # 4-6. Connect to the Gateway and run the agent while the connection is open
        mcp_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))
        with mcp_client:
            gateway_tools = mcp_client.list_tools_sync()
            tools = [
                search_knowledge_base,
                calculate_loyalty_discount,
                get_page_title,
            ]
            tools.extend(gateway_tools)

            agent = Agent(
                model=model,
                tools=tools,
                system_prompt=SYSTEM_PROMPT,
                hooks=[memory_hook],
            )
            response = await agent.invoke_async(user_input)

        # 7. Return the text from the first content block of the response
        try:
            return response.message["content"][0]["text"]
        except (AttributeError, KeyError, IndexError, TypeError):
            return str(response)

    except Exception as e:
        # 8. Handle exceptions gracefully
        logger.error(f"Error handling agent invocation: {e}", exc_info=True)
        return f"I apologize, but I encountered an error processing your request: {str(e)}"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()