# 🧠 Narrative Consistency Verification  
### Kharagpur Data Science Hackathon (KDSH) – Track A

> **Task:** Verify whether a character backstory claim is logically and causally consistent with long-form literary narratives.

This repository presents a retrieval-based semantic consistency solution for **Track A of KDSH**, addressing global consistency challenges in long novels such as *The Count of Monte Cristo* and *In Search of the Castaways*.

---

## 🎯 Problem Statement

Large Language Models often fail to maintain global narrative consistency over long texts.  
This challenge reframes narrative understanding as a **binary classification problem**.

**Input**
- Character name  
- Backstory / claim  
- Source novel  

**Output**
- `1` → Consistent  
- `0` → Inconsistent  

**Key Challenge:** Relevant evidence is sparsely distributed across long documents, requiring effective retrieval rather than full-context generation.

---

## 🧠 Methodology

We adopt an evidence-grounded retrieval pipeline:

1. **Text Chunking**  
   Novels are split into overlapping chunks (~800 characters) to ensure dense, localized context.

2. **Data Ingestion (Pathway – Mandatory)**  
   Structured ingestion of text chunks and claims using the Pathway framework.

3. **Semantic Embeddings**  
   - Model: `all-MiniLM-L6-v2` (384-dim)  
   - Claim embedding = *Character + Backstory*  
   - Corpus = novel text chunks  

4. **Similarity-Based Reasoning**  
   Cosine similarity is computed between claims and all chunks, selecting the best-matching evidence.

5. **Decision Rule (Validated Threshold)**  
   - Similarity **> 0.45 → Consistent (1)**  
   - Similarity **≤ 0.45 → Inconsistent (0)**  

---

## 🧩 Why This Works

- Scales to long narratives  
- Avoids LLM context-window limits  
- Evidence-driven and interpretable  
- Computationally efficient  
- Fully compliant with hackathon constraints  

---

## 🛠️ Tech Stack

- Python  
- Pathway Framework  
- SentenceTransformers (`all-MiniLM-L6-v2`)  
- PyTorch  
- NumPy, Pandas  
- tqdm  

---

## 📂 Repository Structure

