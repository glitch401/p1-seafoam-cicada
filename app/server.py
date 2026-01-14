from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langserve import add_routes
import json, os, re

from app.graph import graph

app = FastAPI(title="Phase 1 Mock API + Chat UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOCK_DIR = os.path.join(ROOT, "mock_data")

def load(name):
    path = os.path.join(MOCK_DIR, name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find {name}")
        return []

ORDERS = load("orders.json")
ISSUES = load("issues.json")
REPLIES = load("replies.json")

class TriageInput(BaseModel):
    ticket_text: str
    order_id: str | None = None

@app.get("/health")
def health(): 
    return {"status": "ok"}

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

@app.post("/classify/issue")
def classify_issue(payload: dict):
    return {"issue_type": "unknown", "confidence": 0.1}

@app.post("/reply/draft")
def reply_draft(payload: dict):
    return {"reply_text": "placeholder"}

@app.post("/triage/invoke")
async def triage_invoke_manual(body: TriageInput):
    # 1. Prepare Inputs for the Graph
    inputs = {"ticket_text": body.ticket_text}
    if body.order_id:
        inputs["order_id"] = body.order_id

    # 2. Run the Agent
    # We use a static thread_id because the assessment script is stateless
    config = {"configurable": {"thread_id": "assessment_mode"}}
    final_state = await graph.ainvoke(inputs, config=config)
    
    # 3. Extract Outputs
    agent_issue = final_state.get("issue_type")
    agent_oid = final_state.get("order_id")
    agent_reply = final_state["messages"][-1].content
    
    # 4. Lookup full order object (Required by template response schema)
    full_order_obj = next((o for o in ORDERS if o["order_id"] == agent_oid), None)

    # 5. Return Specific JSON Format
    return {
        "order_id": agent_oid,
        "issue_type": agent_issue,
        "order": full_order_obj,
        "reply_text": agent_reply
    }


add_routes(
    app,
    graph,
    path="/agent",
    playground_type="default"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)