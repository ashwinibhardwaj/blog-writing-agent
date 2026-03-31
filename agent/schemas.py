from pydantic import BaseModel, Field
from typing import List, Literal, Optional



class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(description = "One sentence describing what the reader should be able to do/understand after this section. ",)
    bullets: List[str] = Field(
        min_length=3,
        max_length=6,
        description = "3-6 concrete, non-overlapping subpoints to cover this section. ",
    )

    target_words: int = Field(description="Target word count for this section.",)
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False




class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: Literal["technical/deep-dive","beginner-friendly","opinion/hot-take","tutorial/how-to"] = "technical/deep-dive"

    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: str = ""
    snippet: str = ""
    source: str = ""


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "open_book", "hybrid"]
    reason: set
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5, description="how many results to fetch per query(3-5).")

class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)