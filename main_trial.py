import pandas as pd
import pathway as pw
from sentence_transformers import SentenceTransformer
import openai
import os

# ================= CONFIG =================
openai.api_key = "YOUR_API_KEY"   # submit time pe env variable use karna

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# =========================================
# LOAD CSV
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")


# =========================================
# LOAD BOOK TEXT
def load_book(book_name):
    if book_name == "In Search of the Castaways":
        path = "data/In Search of the Castaways.txt"
    elif book_name == "The Count of Monte Cristo":
        path = "data/The Count of Monte Cristo.txt"
    else:
        raise ValueError("Unknown book")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# =========================================
# CHUNK TEXT
def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# =========================================
# PATHWAY VECTOR INDEX
def build_pathway_index(chunks):
    df = pd.DataFrame({"text": chunks})
    table = pw.debug.table_from_pandas(df)
    index = pw.ml.index.KNNIndex(
        table.text,
        embedder=lambda x: EMBED_MODEL.encode(x)
    )
    return index


# =========================================
# RETRIEVE EVIDENCE
def retrieve_evidence(index, character, claim, k=5):
    query = f"{character}. {claim}"
    results = index.query(query, k=k)
    return [r[0] for r in results]


# =========================================
# LLM CONSISTENCY CHECK
def llm_consistency_check(evidence, claim):
    prompt = f"""
You are checking consistency between a novel and a backstory claim.

CLAIM:
{claim}

EVIDENCE FROM NOVEL:
{evidence}

Question:
Does the evidence SUPPORT or CONTRADICT the claim?

Answer with only one word:
SUPPORT or CONTRADICT
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response["choices"][0]["message"]["content"].strip().lower()

    if "support" in answer:
        return 1
    else:
        return 0


# =========================================
# MAIN LOOP
results = []

for idx, row in test_df.iterrows():
    sample_id = row[0]
    book_name = row[1]
    character = row[2]
    claim = row[4]

    # Load + index book
    book_text = load_book(book_name)
    chunks = chunk_text(book_text)
    index = build_pathway_index(chunks)

    # Retrieve evidence
    evidence_chunks = retrieve_evidence(index, character, claim)
    evidence_text = "\n\n".join(evidence_chunks)

    # LLM reasoning
    prediction = llm_consistency_check(evidence_text, claim)

    results.append({
        "id": sample_id,
        "prediction": prediction
    })

    if idx < 2:
        print("ID:", sample_id)
        print("Prediction:", prediction)
        print("-" * 50)


# ========== SAVE RESULTS ==========
results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)

print("✅ results.csv generated (SUBMIT THIS)")

print("\n✅ DONE. train_predictions.csv generated.")
