import os
import sys
import uuid
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = "hr-policy-index-v2"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

def ingest_policy_document(file_path: str):
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY is missing in .env")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        pc = Pinecone(api_key=api_key)

        # 1. Automatically check and create a standard vector index if it doesn't exist
        existing_indexes = [idx["name"] for idx in pc.list_indexes()]
        if INDEX_NAME not in existing_indexes:
            print(f"[*] Creating standard vector index '{INDEX_NAME}'...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"[+] Index '{INDEX_NAME}' created successfully.")
        else:
            print(f"[*] Index '{INDEX_NAME}' already exists.")

        print(f"[*] Reading document: {file_path}...")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        if not raw_text.strip():
            raise ValueError("The file is empty!")

        print("[*] Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = text_splitter.create_documents([raw_text])
        print(f"[+] Generated {len(chunks)} chunks.")

        print(f"[*] Generating embeddings with {EMBEDDING_MODEL_NAME}...")
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        texts = [chunk.page_content for chunk in chunks]
        vectors = embeddings_model.embed_documents(texts)

        print(f"[*] Uploading to Pinecone index '{INDEX_NAME}'...")
        index = pc.Index(INDEX_NAME)

        records = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            records.append({
                "id": f"doc_{uuid.uuid4().hex[:8]}_{i}",
                "values": vector,
                "metadata": {
                    "text": chunk.page_content,
                    "source": file_path
                }
            })

        batch_size = 100
        for i in range(0, len(records), batch_size):
            index.upsert(vectors=records[i:i + batch_size])

        print("[SUCCESS] Ingestion completed! Vectors are securely stored in Pinecone.")

    except Exception as e:
        print(f"[ERROR] Ingestion failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    ingest_policy_document("data/leave_policy.pdf")