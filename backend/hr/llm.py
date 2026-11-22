import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Fast model for Tool Selection, Classification & Grading
fast_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# Smart model for Final Answer, Decomposition & SQL Generation
smart_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)
