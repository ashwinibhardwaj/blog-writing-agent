import operator
from typing import TypedDict, List, Optional, Annotated
from agent.schemas import EvidenceItem, Plan

class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    as_of: str 
    recency_days: int

    sections: Annotated[List[tuple[int, str]], operator.add]
    final: str