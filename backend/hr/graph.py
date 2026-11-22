from langgraph.graph import END, StateGraph
from .nodes import (
    rewrite_node,
    planner_node,
    tool_selector_node,
    retrieve_node,
    sql_node,
    calculator_node,
    web_node,
    generate_node,
)
from .state import GraphState

# 1. Initialize the StateGraph with our GraphState schema
workflow = StateGraph(GraphState)

# 2. Add all workflow nodes
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("planner", planner_node)
workflow.add_node("tool_selector", tool_selector_node)
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("sql_node", sql_node)
workflow.add_node("calculator_node", calculator_node)
workflow.add_node("web_node", web_node)
workflow.add_node("generate_node", generate_node)

# 3. Entry point: start at rewrite
workflow.set_entry_point("rewrite")

# 4. Pipeline flow: rewrite -> planner -> tool_selector
workflow.add_edge("rewrite", "planner")
workflow.add_edge("planner", "tool_selector")

# 5. Conditional routing from tool_selector
def route_decision(state: GraphState):
    route = state.get("route", "retrieve")
    if route == "sql":
        return "sql_node"
    elif route == "calculator":
        return "calculator_node"
    elif route == "web":
        return "web_node"
    elif route == "retrieve":
        return "retrieve_node"
    else:
        return "generate_node"

workflow.add_conditional_edges(
    "tool_selector",
    route_decision,
    {
        "sql_node": "sql_node",
        "retrieve_node": "retrieve_node",
        "calculator_node": "calculator_node",
        "web_node": "web_node",
        "generate_node": "generate_node",
    },
)

# 6. Tool nodes connect to generate_node
workflow.add_edge("sql_node", "generate_node")
workflow.add_edge("retrieve_node", "generate_node")
workflow.add_edge("calculator_node", "generate_node")
workflow.add_edge("web_node", "generate_node")

# 7. Final generate_node terminates at END
workflow.add_edge("generate_node", END)

# 8. Compile the LangGraph runnable application
app_graph = workflow.compile()
