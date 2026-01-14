from typing import TypedDict, Annotated, List, Optional, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from enum import Enum
from app.utils import get_issue_types

raw_issues = get_issue_types()
enum_map = {item.upper(): item for item in raw_issues}
IssueType = Enum('IssueType', enum_map)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Context fields
    ticket_text: Optional[str]
    issue_type: Optional[IssueType]
    order_id: Optional[str]
    evidence: Optional[str]

class ClassificationOutput(BaseModel):
    issue_type: IssueType = Field(..., description="""
    Classify the issue strictly into one of the valid categories.
    
    CRITICAL DEFINITIONS:
    - 'late_delivery': Package has not arrived yet, is delayed, or user asking for status.
    - 'missing_item': Package arrived but a specific product is missing inside.
    - 'defective_product': Item arrived but is not working.
    - 'damaged_item': Item arrived physically broken.
    """)
    
    order_id: Optional[str] = Field(
        None, description="The alpha-numeric order ID. None if missing."
    )
    # evidence: str = Field(..., description="Quote justifying the classification.")