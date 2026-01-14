import json
from pathlib import Path
from typing import Optional, Dict, Any, List

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "mock_data"

def load_json(filename: str) -> List[Dict]:
    file_path = DATA_DIR / filename
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found at {file_path}")
        return []

def lookup_order(order_id: str) -> Optional[Dict[str, Any]]:
    if not order_id: 
        return None
    
    db = load_json("orders.json")
    clean_id = str(order_id).replace('-', '').replace(' ', '').upper()

    for order in db:
        db_id = str(order.get('order_id', '')).replace('-', '').replace(' ', '').upper()
        if db_id == clean_id:
            return order
    return None

def get_reply_template(issue_type: str) -> str:
    replies = load_json("replies.json")
    for item in replies:
        if item.get("issue_type") == issue_type:
            return item["template"]
    return "Hi {{customer_name}}, regarding order {{order_id}}: We are looking into your issue."

def get_issue_types()->List[str]:
    issue_mappings = load_json("issues.json")
    return [item["issue_type"] for item in issue_mappings]+['other']