import os
import uuid
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv("backend/.env")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

HR_POLICY_DOCUMENTS = [
    {
        "source": "Leave policy Handbook.pdf",
        "title": "Leave and Time-Off Policy Handbook",
        "content": """
HR Leave Policy Handbook - Chapter 1: Time Off & Leave Allocations

1. Paid Sick Leave Policy:
Employees receive 80 Hours Paid Sick Leave annually. The leave balance is granted on January 1 of each year. New hires receive a prorated allocation based on their official start date. Sick leave covers employee illness, preventive medical care, medical emergencies, mental health days, and care for immediate family members. Unused sick leave does not convert to cash at termination.

2. Annual Vacation and Casual Leave:
All full-time staff accrue 15 days of paid annual leave per year, accruing at 1.25 days per month of service. Employees may carry forward up to 5 unused annual leave days into the subsequent calendar year.

3. Work From Home (WFH) and Remote Work:
Eligible employees may utilize hybrid remote work up to 2 days per week, subject to departmental supervisor approval. Remote employees are expected to maintain core availability between 9:00 AM and 5:00 PM.

4. Parental and Family Leave:
Eligible employees are entitled to 12 consecutive weeks of fully paid parental leave for the birth, adoption, or foster placement of a child. Bereavement leave provides up to 5 consecutive paid days off for the loss of an immediate family member.
"""
    },
    {
        "source": "Benefits Policy Handbook.pdf",
        "title": "Employee Benefits and Welfare Policy",
        "content": """
HR Benefits Policy Handbook - Chapter 2: Health, Retirement & Wellness

1. Comprehensive Health, Dental & Vision Insurance:
The organization provides premium health, dental, and vision insurance for all full-time employees and eligible dependents. Coverage begins on the first calendar day of the month following the hire date. The company subsidizes 85% of employee healthcare premiums and 60% of dependent coverage.

2. 401(k) Retirement Savings & Employer Matching:
Employees may contribute pre-tax earnings to the organizational 401(k) plan immediately upon hire. The company matches 100% of the first 3% contributed, and 50% of the next 2% (totaling a maximum company match of 4% of base salary). Matching vests immediately.

3. Disability and Life Insurance:
Short-term disability covers up to 66% of base wages for up to 26 weeks after an 8-day waiting period. Long-term disability and basic life insurance (equal to 2x annual base salary) are provided at zero cost to employees.
"""
    },
    {
        "source": "Compensation Guide.docx",
        "title": "Compensation and Remuneration Guidelines",
        "content": """
Compensation Guide & Remuneration Structure - Chapter 3: Pay and Incentives

1. Compensation Definition and Structure:
Compensation represents direct monetary remuneration provided to employees for work performed. It includes:
- Base Salary / Hourly Wages: Guaranteed monetary compensation established by market benchmarks.
- Annual Performance Bonus: Discretionary cash incentives awarded annually based on individual KPIs and company profitability.
- Sales Commissions & Overtime: Direct incentive payouts tied directly to quota achievement and hours worked beyond standard 40-hour workweeks.

2. Comparison Summary - Compensation versus Benefits:
- Compensation: Focuses on direct monetary remuneration including base salary, annual performance bonuses, overtime pay, and sales commission structures.
- Benefits: Encompasses indirect non-monetary value including health/dental insurance, 401(k) matching, paid time off, and disability coverage.
Both components together constitute Total Rewards, ensuring fair financial rewards alongside long-term well-being and health security.
"""
    }
]

def seed_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("PINECONE_API_KEY missing")
        return

    pc = Pinecone(api_key=api_key)
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    target_indexes = ["hr-policy-index-v2", "hr-policy-index"]
    for idx_name in target_indexes:
        if idx_name not in existing_indexes:
            print(f"Creating index {idx_name}...")
            pc.create_index(
                name=idx_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

    print("Loading embedding model...")
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)

    all_records = []
    for doc in HR_POLICY_DOCUMENTS:
        chunks = splitter.create_documents([doc["content"]])
        texts = [chunk.page_content for chunk in chunks]
        vectors = embeddings_model.embed_documents(texts)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            all_records.append({
                "id": f"{doc['source'].replace(' ', '_').lower()}_{i}_{uuid.uuid4().hex[:6]}",
                "values": vector,
                "metadata": {
                    "text": chunk.page_content,
                    "source": doc["source"],
                    "title": doc["title"]
                }
            })

    print(f"Prepared {len(all_records)} chunks across {len(HR_POLICY_DOCUMENTS)} documents.")

    for idx_name in target_indexes:
        print(f"Upserting to {idx_name}...")
        index = pc.Index(idx_name)
        batch_size = 50
        for i in range(0, len(all_records), batch_size):
            index.upsert(vectors=all_records[i:i + batch_size])
        print(f"Successfully upserted into {idx_name}.")

if __name__ == "__main__":
    seed_pinecone()
