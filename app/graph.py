import json, re
from datetime import datetime
from typing import Literal
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.schema import AgentState, ClassificationOutput
from app.utils import lookup_order, get_reply_template

logger = logging.getLogger("uvicorn.error")

model = ChatOllama(model="qwen3:4b", temperature=0, keep_alive=-1)

@tool
def fetch_order_tool(order_id: str) -> str:
    """Retrieves order details from internal ERP system."""
    order = lookup_order(order_id)
    if order:
        return json.dumps(order)
    return f"Order ID {order_id} not found in the database."


def classify_node(state: AgentState):
    """Analyzes conversation history to determine issue and order ID."""
    
    text_to_analyze = state.get("ticket_text")
    if not text_to_analyze and state.get("messages"):
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                text_to_analyze = msg.content
                break
    
    system_prompt = """
    You are an expert Customer Support Triage Agent.
    Your job is to classify the user's issue into exactly one of these categories:

    CLASSIFICATION RULES:
    - 'defective_product': The item arrived but is NOT WORKING, won't turn on, is glitchy, or has a bad battery.
    - 'damaged_item': The item arrived physically BROKEN, smashed, scratched, or crushed.
    - 'missing_item': The order arrived but a specific item was NOT in the box.
    - 'late_delivery': The order has not arrived yet, or the user is asking for shipping status.
    - 'wrong_item': The user received a product they did not order.
    - 'refund_request': The user explicitly asks for their money back.
    - 'duplicate_charge': The user sees two charges for the same order.
    
    Extract the Order ID if present (e.g., ORD-123).
    """
    
    messages = [SystemMessage(content=system_prompt)] + state['messages']

    structured_llm = model.with_structured_output(ClassificationOutput)
    response = structured_llm.invoke(messages)

    issue_type_str = response.issue_type.value if hasattr(response.issue_type, 'value') else response.issue_type

    oid = response.order_id
    if oid:
        clean_oid = oid.strip().upper()
        if any(x in clean_oid for x in ["N/A", "NONE", "NULL", "NOT PROVIDED", "UNKNOWN"]):
            oid = None
            
    if not oid and text_to_analyze:
        match = re.search(r"(ORD[-\s]?\d+)", text_to_analyze, re.IGNORECASE)
        if match:
            oid = match.group(1).upper().replace(" ", "").replace("-", "")

    if not oid:
        oid = state.get("order_id")

    if logger:
        logger.info(f"🧐 CLASSIFIER DECISION:")
        logger.info(f"   -> Issue Type: {issue_type_str}")
        logger.info(f"   -> Order ID:   {oid} (Persisted or New)")

    return {
        'issue_type': issue_type_str,
        'order_id': oid,
        # 'evidence': response.evidence
    }

def fetch_order_node(state: AgentState):
    """Executes the tool and appends result to history."""
    order_id = state.get('order_id')
    
    tool_result = fetch_order_tool.invoke(str(order_id))
    
    return {
        "messages": [
            ToolMessage(
                content=str(tool_result),
                name='fetch_order',
                tool_call_id=f"call_{order_id}"
            )
        ]
    }

def draft_reply_node(state: AgentState):
    """Generates the final response based on context."""

    issue_type = state.get("issue_type")
    order_id = state.get("order_id")
    
    user_input = "Unknown"
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if not order_id:
        prompt = f"""
        System: You are a support agent. User has a '{issue_type}' issue but NO Order ID.
        Context: {user_input}
        Task: Ask for the Order ID politely. Do not promise a solution yet.
        Today's Date: {today}
        """
    else:
        template = get_reply_template(issue_type) if order_id else "N/A"
        prompt = f"""
        You are a helpful Customer Support Agent.
        
        CONTEXT:
        - Issue Type: {issue_type}
        - Order ID: {order_id or "Not Provided"}
        - Standard Company Response: "{template}"
        
        INSTRUCTIONS:
        1. If the user is reporting the issue for the first time, use the "Standard Company Response" (fill in placeholders).
        2. If the user is asking a FOLLOW-UP question (e.g., "When?", "How long?", "Is it free?"), ANSWER the question directly. Do NOT repeat the standard response.
        3. If the user asks for a refund timeline, say "within 5 business days".
        4. If the user asks about replacement, say "A replacement will be arranged under warranty."
        5. If the user asks about returning a wrong item, say something like: "We will send a prepaid return label and ship the correct item."
        6. If the Order ID was found, MENTION the item name to confirm you found the right order.
        Today's Date: {today}
        """
    messages = [SystemMessage(content=prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# --- ROUTING ---

def route_ticket(state: AgentState) -> Literal['fetch_order', 'draft_reply']:
    if state.get('order_id'):
        return 'fetch_order'
    return 'draft_reply'

# --- GRAPH CONSTRUCTION ---

builder = StateGraph(AgentState)

builder.add_node('classify', classify_node)
builder.add_node('fetch_order', fetch_order_node)
builder.add_node('draft_reply', draft_reply_node)

builder.add_edge(START, 'classify')

builder.add_conditional_edges(
    'classify', 
    route_ticket, 
    {'fetch_order': 'fetch_order', 'draft_reply': 'draft_reply'}
)

builder.add_edge('fetch_order', 'draft_reply')
builder.add_edge('draft_reply', END)

# MEMORY: This is what enables multi-turn conversation
# memory = MemorySaver()

graph = builder.compile(
    #checkpointer=memory #The LangGraph CLI handles memory automatically, so it forbids you from manually adding a checkpointer in your code.
    )