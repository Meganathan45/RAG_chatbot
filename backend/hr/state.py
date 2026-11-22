from typing import List, TypedDict, Dict, Any, Optional

class GraphState(TypedDict, total=False):
    """Represents the complete state of our Agentic RAG graph."""
    question: str
    standalone_query: Optional[str]
    plan: Optional[List[str]]
    steps: Optional[List[Dict[str, Any]]]
    route: Optional[str]  # 'retrieve', 'sql', 'calculator', 'web', 'chat'
    sub_queries: Optional[List[str]]
    documents: Optional[List[str]]
    sources: Optional[List[str]]
    grade: Optional[str]
    sql_query: Optional[str]
    sql_result: Optional[str]
    math_result: Optional[str]
    web_result: Optional[str]
    generation: Optional[str]
