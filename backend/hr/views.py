import traceback
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Employee
from .serializers import EmployeeSerializer
from .graph import app_graph

def home_view(request):
    """Serve the frontend application directly from Django."""
    return render(request, 'home.html')

@api_view(['GET'])
def health_check(request):
    """Simple health check endpoint."""
    return Response({
        "status": "healthy",
        "service": "Agentic RAG HR-Verse Backend", 
        "version": "2.5 Hybrid"
    })

@api_view(['GET', 'POST'])
def employee_list_create(request):
    """List or create employees in SQLite."""
    if request.method == 'GET':
        employees = Employee.objects.all()
        serializer = EmployeeSerializer(employees, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = EmployeeSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def chat_view(request):
    """
    Main Agentic RAG Chat endpoint.
    Accepts: { "question": "..." }
    Returns: {
        "answer": "...",
        "tool": "...",
        "plan": [...],
        "steps": [...],
        "sources": [...],
        "sql_query": "..."
    }
    """
    question = request.data.get('question', '').strip()
    if not question:
        return Response(
            {"error": "Question parameter is required."},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        # Run state through LangGraph agent workflow
        initial_state = {"question": question}
        final_state = app_graph.invoke(initial_state)

        answer = final_state.get("generation", "No response generated.")
        tool = final_state.get("route", "chat")
        plan = final_state.get("plan", [])
        steps = final_state.get("steps", [])
        sources = final_state.get("sources", [])
        sql_query = final_state.get("sql_query")
        sub_queries = final_state.get("sub_queries", [question])

        return Response({
            "answer": answer,
            "tool": tool,
            "plan": plan,
            "steps": steps,
            "sources": sources,
            "sql_query": sql_query,
            "subQueries": sub_queries
        }, status=status.HTTP_200_OK)

    except Exception as e:
        traceback.print_exc()
        return Response({
            "error": str(e),
            "trace": traceback.format_exc()
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
