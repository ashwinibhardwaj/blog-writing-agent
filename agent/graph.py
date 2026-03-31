from langgraph.graph import StateGraph, START, END
from agent.nodes import router_node, route_next, research_node, orchestrator_node, fanout, worker_node, reducer_node
from agent.state import State
from agent.schemas import Plan
from typing import Optional
from datetime import date

g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_node)


g.add_edge(START,"router")
g.add_conditional_edges("router", route_next,
                       {
                           "research":"research",
                           "orchestrator": "orchestrator"
                       })
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout,["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

blog_app = g.compile()



def run(topic: str, as_of: Optional[str] = None):
    if as_of is None:
        as_of = date.today().isoformat()

    out = blog_app.invoke(
        {
            "topic": topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "as_of": as_of,
            "recency_days": 7,   # router may overwrite
            "sections": [],
            "final": "",
        }
    )

    plan: Plan = out["plan"]
    print("\n" + "=" * 100)
    print("TOPIC:", topic)
    print("AS_OF:", out.get("as_of"), "RECENCY_DAYS:", out.get("recency_days"))
    print("MODE:", out.get("mode"))
    print("BLOG_KIND:", plan.blog_kind)
    print("NEEDS_RESEARCH:", out.get("needs_research"))
    print("QUERIES:", (out.get("queries") or [])[:6])
    print("EVIDENCE_COUNT:", len(out.get("evidence", [])))
    if out.get("evidence"):
        print("EVIDENCE_SAMPLE:", [e.model_dump() for e in out["evidence"][:2]])
    print("TASKS:", len(plan.tasks))
    print("SAVED_MD_CHARS:", len(out.get("final", "")))
    print("=" * 100 + "\n")

    

    return out



