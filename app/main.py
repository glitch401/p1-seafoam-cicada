from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json, os, re
import uuid

from app.graph import graph
from app.utils import lookup_order

app = FastAPI(title="Phase 1 Mock API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- BOILERPLATE DATA LOADING (Kept for compatibility) ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOCK_DIR = os.path.join(ROOT, "mock_data") # Or "data", depending on your folder setup

def load(name):
    # Robust path finding
    paths = [
        os.path.join(MOCK_DIR, name),
        os.path.join(ROOT, "data", name),
        os.path.join(ROOT, "..", "data", name)
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return [] # Return empty if not found to prevent crash

ORDERS = load("orders.json")
ISSUES = load("issues.json")
REPLIES = load("replies.json")

class TriageInput(BaseModel):
    ticket_text: str
    order_id: str | None = None
    # Add an optional conversation_id if the client supports it
    conversation_id: str | None = None 

# --- HELPER ENDPOINTS (Required by Boilerplate) ---

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/orders/get")
def orders_get(order_id: str = Query(...)):
    for o in ORDERS:
        if o["order_id"] == order_id: return o
    raise HTTPException(status_code=404, detail="Order not found")

@app.get("/orders/search")
def orders_search(customer_email: str | None = None, q: str | None = None):
    matches = []
    for o in ORDERS:
        if customer_email and o["email"].lower() == customer_email.lower():
            matches.append(o)
        elif q and (o["order_id"].lower() in q.lower() or o["customer_name"].lower() in q.lower()):
            matches.append(o)
    return {"results": matches}

# Legacy endpoints (Stubbed)
@app.post("/classify/issue")
def classify_issue(payload: dict):
    return {"issue_type": "unknown", "confidence": 0.1}

@app.post("/reply/draft")
def reply_draft(payload: dict):
    return {"reply_text": "placeholder"}


# --- THE SMART AI ENDPOINT ---

@app.post("/triage/invoke")
async def triage_invoke(body: TriageInput):
    """
    Executes the Unified Graph with Memory Persistence.
    """
    ticket_text = body.ticket_text
    explicit_oid = body.order_id
    
    # 1. DETERMINE THREAD ID (The "Memory" Key)
    # Strategy:
    # A. If client sends 'conversation_id', use that (Best for Chat UI).
    # B. If client sends 'order_id', use that (Good for Assessment).
    # C. If neither, we generate a random one (Stateless fallback).
    
    thread_id = body.conversation_id
    if not thread_id and explicit_oid:
        thread_id = explicit_oid
    if not thread_id:
        # Try to find ID in text to use as thread anchor
        m = re.search(r"(ORD[-\s]?\d+)", ticket_text, re.IGNORECASE)
        if m:
            thread_id = m.group(1).upper().replace(" ", "").replace("-", "")
    
    if not thread_id:
        thread_id = str(uuid.uuid4()) # Fallback: No memory persistence across calls if no ID found

    # 2. PREPARE GRAPH INPUT
    inputs = {
        "ticket_text": ticket_text,
        "messages": [("user", ticket_text)]
    }
    if explicit_oid:
        inputs["order_id"] = explicit_oid

    # 3. RUN GRAPH (Async)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        final_state = await graph.ainvoke(inputs, config=config)
    except Exception as e:
        # Fallback for unexpected AI errors
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")

    # 4. FORMAT OUTPUT (Matching the Boilerplate Schema)
    
    # Extract Reply
    # The unified node puts the final reply as the last AI message
    last_msg = final_state["messages"][-1]
    reply_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    
    # Extract Metadata
    agent_issue = final_state.get("issue_type", "other")
    agent_oid = final_state.get("order_id")
    
    # Hydrate Order Object (Required by Boilerplate)
    full_order = None
    if agent_oid:
        full_order = next((o for o in ORDERS if o["order_id"] == agent_oid), None)

    return {
        "order_id": agent_oid,
        "issue_type": agent_issue,
        "order": full_order,
        "reply_text": reply_text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)