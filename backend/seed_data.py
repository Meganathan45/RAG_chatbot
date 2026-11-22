"""
Seed Script: Initial Database & Knowledge Base Setup
Runs SQLite employee table migrations, seeds 5 staff members,
creates the SQL compatibility view, and initializes Pinecone vectors.
"""
import os
import sys
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

# Ensure backend path is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

def seed_sqlite():
    db_path = BASE_DIR / "db.sqlite3"
    print(f"[*] Seeding SQLite database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Ensure table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hr_employee (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL,
            department VARCHAR(100) NOT NULL,
            salary REAL NOT NULL
        );
    """)

    # Seed 5 sample employees
    cur.execute("DELETE FROM hr_employee;")
    sample_employees = [
        (1, 'Alice Johnson', 'Engineering', 95000.0),
        (2, 'Bob Smith', 'Human Resources', 68000.0),
        (3, 'Charlie Brown', 'Sales', 72000.0),
        (4, 'Diana Prince', 'Marketing', 74000.0),
        (5, 'Evan Wright', 'Engineering', 105000.0),
    ]
    cur.executemany("INSERT INTO hr_employee (id, name, department, salary) VALUES (?, ?, ?, ?);", sample_employees)

    # Create SQL view alias for table 'employees'
    cur.execute("CREATE VIEW IF NOT EXISTS employees AS SELECT * FROM hr_employee;")
    conn.commit()

    count = cur.execute("SELECT COUNT(id) FROM hr_employee;").fetchone()[0]
    conn.close()
    print(f"[+] Successfully seeded {count} employees into 'hr_employee' and created 'employees' view.")

def seed_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("[-] Skipping Pinecone setup: PINECONE_API_KEY is not set in backend/.env")
        return

    try:
        from pinecone import Pinecone, ServerlessSpec
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        INDEX_NAME = "hr-policy-index-v2"
        pc = Pinecone(api_key=api_key)
        existing = [idx["name"] for idx in pc.list_indexes()]

        if INDEX_NAME not in existing:
            print(f"[*] Creating Pinecone vector index '{INDEX_NAME}'...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"[+] Index '{INDEX_NAME}' created.")
        else:
            print(f"[*] Pinecone index '{INDEX_NAME}' is ready.")

        # Ingest rich HR policy documents if not populated
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        if stats.get("total_vector_count", 0) < 5:
            print("[*] Embedding and upserting HR policy knowledge base...")
            from seed_pinecone import HR_POLICY_DOCUMENTS
            emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
            
            records = []
            import uuid
            for doc in HR_POLICY_DOCUMENTS:
                chunks = splitter.create_documents([doc["content"]])
                texts = [c.page_content for c in chunks]
                vecs = emb.embed_documents(texts)
                for i, (c, v) in enumerate(zip(chunks, vecs)):
                    records.append({
                        "id": f"{doc['source'].replace(' ', '_').lower()}_{i}_{uuid.uuid4().hex[:6]}",
                        "values": v,
                        "metadata": {"text": c.page_content, "source": doc["source"]}
                    })
            for i in range(0, len(records), 50):
                index.upsert(vectors=records[i:i+50])
            print(f"[+] Upserted {len(records)} chunks into '{INDEX_NAME}'.")
        else:
            print(f"[+] Pinecone index '{INDEX_NAME}' already contains {stats.get('total_vector_count')} vectors.")

    except Exception as e:
        print(f"[-] Pinecone seeding notice: {e}")

if __name__ == "__main__":
    seed_sqlite()
    seed_pinecone()
    print("[SUCCESS] Environment initialization completed.")
