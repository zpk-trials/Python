<details>
   <summary><b>The Minimal Version Best for Hack-a-Thon (Choice - A)</b></summary>

# Kharagpur Data Science Hackathon (KDSH) – Track A

> **Task:** Verify whether a character backstory claim is logically and causally consistent with long-form literary narratives.

This repository presents a retrieval-based semantic consistency solution for **Track A of the Kharagpur Data Science Hackathon (KDSH)**, addressing global consistency challenges in long novels such as *The Count of Monte Cristo* and *In Search of the Castaways*.

---

## 🎯 Problem Statement

Large Language Models often fail to maintain global narrative consistency over long texts.  
This challenge reframes narrative understanding as a **binary classification problem**.

### Input
- Character name  
- Backstory / claim  
- Source novel  

### Output
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
   - Model: all-MiniLM-L6-v2 (384-dimensional)  
   - Claim embedding = Character + Backstory  
   - Corpus = novel text chunks  

4. **Similarity-Based Reasoning**  
   Cosine similarity is computed between claims and all chunks, selecting the best-matching evidence.

5. **Decision Rule (Validated Threshold)**  
   - Similarity > 0.45 → Consistent (1)  
   - Similarity ≤ 0.45 → Inconsistent (0)

---

## 🧩 Why This Works

- Scales to long narratives  
- Avoids LLM context-window limitations  
- Evidence-driven and interpretable  
- Computationally efficient  
- Fully compliant with hackathon constraints  

---

## 🛠️ Tech Stack

- Python  
- Pathway Framework  
- SentenceTransformers (all-MiniLM-L6-v2)  
- PyTorch  
- NumPy  
- Pandas  
- tqdm  

---

<details>
   <summary><b><H3>📂 Repository Structure</H3></b></summary>  
&ensp;<b>└── README.md</b>
 <details>
    <summary><b>└── Code </b></summary>
&emsp;   ├── final.py<br>
&emsp;   ├── train.csv<br>
&emsp;   ├── test.csv<br>
&emsp;   ├── In search of the castaways.txt<br>
&emsp;   ├── The Count of Monte Cristo.txt  <br>
&emsp;   └── results.csv  <br>
 </details>


</details>
---

## ▶️ Quick Start

Install dependencies  
```bash
pip install -r requirements.txt
```

Run the solution  
```bash
python final.py
```

**Output:**  
Generates `results.csv` containing binary consistency predictions.

---

## 🧪 Core Insight

Consistent claims exhibit strong semantic alignment with at least one specific passage in the novel.  
Inconsistent claims lack sufficient semantic evidence or contradict the narrative.

---

## 🏁 Submission

Track: A  
Hackathon: Kharagpur Data Science Hackathon 2025

```
 _   __  ____    ____  _   _
| |/ / |  _ \  / ___|| | | |
| ' /  | | | | \___ \| |_| |
| . \  | |_| |  ___) |  _  |
|_|\_\ |____/  |____/|_| |_|
```

| Team-mates | Contributions |
| :--- | :--- |
| Vinayak Dhiman | int |
| Shaurya Swaraj | str |
| Diksha Jangra | float |
| Pritham Prajwin V | Double |

</details>

<details>
   <summary><b>The One In Detail (Choice - B)</summary>

# Kharagpur Data Science Hackathon - Narrative Consistency (Track A)

This repository contains the solution for Track A of the Kharagpur Data Science Hackathon. The objective is to determine whether a specific backstory or claim about a character is causally and logically consistent with the events in long-form narratives (The Count of Monte Cristo and In Search of the Castaways).

## 📌 Problem Statement

Large Language Models often struggle with global consistency over long narratives. This challenge treats narrative consistency as a structured classification problem:

- Input: A character, a specific claim/backstory, and the source novel.

- Task: Determine if the claim is consistent (`1`) or inconsistent (`0`) with the book.

- Constraint: The evidence is distributed across long contexts requiring retrieval and semantic matching.

---

## 🛠️ Approach & Methodology

Our solution utilizes a Retrieval-Based Semantic Consistency approach. Instead of feeding the entire novel into a generative LLM (which is computationally expensive and prone to context-window loss), we use vector embeddings to find semantic alignment between the claim and specific segments of the text.

### Key Components

1. Data Ingestion (`pathway`): We utilize Pathway for efficient data handling and table creation from the raw text chunks.

2. Text Chunking:
The full novels are split into sliding windows of 800 characters. This ensures that the context provided to the embedding model is dense and specific, reducing noise from irrelevant chapters.

3. Vector Embeddings (`sentence-transformers`):
We use the `all-MiniLM-L6-v2` model. This model maps sentences and paragraphs to a 384-dimensional dense vector space.

- Query: The character name + the claim (e.g., "Faria. He spent 4 years writing treatises...")

- Corpus: The chunks of the specific novel.

4. Similarity Scoring & Classification:

- We calculate the **Cosine Similarity** between the Claim Vector and all Book Chunk Vectors.

- We identify the "Best Match" (highest similarity score).

- Thresholding: Based on validation against the training set, we established a decision boundary of 0.45.

  - Score > 0.45 $\rightarrow$ **Consistent (1)** (High semantic overlap found).

  - Score $\le$ 0.45 $\rightarrow$ **Inconsistent (0)** (No sufficient evidence found in text).

---

## 📂 Repository Structure
```
├── In search of the castaways.txt   # Source Text 1
├── The Count of Monte Cristo.txt    # Source Text 2
├── final.py                         # Main execution script
├── requirements.txt                 # Python dependencies
├── test.csv                         # Input test data
├── train.csv                        # Training data (for reference)
└── README.md                        # Documentation
```


---

## 🚀 Setup and Execution

To ensure reproducibility in a clean environment, follow these steps:

**1. Prerequisites**

Ensure you have Python 3.8+ installed.

**2. Install Dependencies**

Install the required libraries using the provided requirements file:

```bash
pip install -r requirements.txt
```

**3. Run the Solution**

Execute the main script. This will load the models, process the text files, and generate the predictions.

```bash
python final.py
```

**4. Output**

The script will generate a file named 'results.csv' in the root directory containing the predictions for the test set.

---

## 📊 Logic & Reasoning

The core hypothesis of this solution is that Consistent claims will have a high semantic resemblance to at least one specific passage in the book (e.g., a claim about "Faria writing treatises" will vector-match closely with the book paragraph describing Faria's writing). Inconsistent claims will either be hallucinations or direct contradictions, resulting in low similarity scores across all chunks.

---
## 📦 Dependencies
- `pathway`

- `sentence-transformers`

- `pandas`

- `numpy`

- `torch`

- `tqdm`

---
# Submitted for Kharagpur Data Science Hackathon 2025.
```
 _   __  ____    ____  _   _ 
| |/ / |  _ \  / ___|| | | |
| ' /  | | | | \___ \| |_| |
| . \  | |_| |  ___) |  _  |
|_|\_\ |____/  |____/|_| |_|

```
---

</details>
