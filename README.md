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






---

## 4. Coding & Technical Blocks
To make code look "boxed" and readable, use **Backticks**:

* **Inline Code:** Use single backticks `` `code` `` for variables like `path/to/file`.
* **Code Blocks with Syntax Highlighting:**
    Always specify the language (python, bash, json) for better colors:
    ```python
    def hello():
        print("Hello World")
    ```

---

## 5. Modern "Extra" Features
### Emojis
You can use shortcodes like `:rocket:` 🚀 or `:warning:` ⚠️. They are essential for breaking up walls of text and making the README feel modern.

### Badges (The "Professional" Look)
Add custom status badges using [Shields.io](https://shields.io):
```markdown
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
```
```diff
+ This line is green
- This line is red
! This line is orange
? tfwece
* cdedwqwev
> 1233fvwe
$ vfeqcf3v43
# vewcf4
```


| Parameter | Type | Description |
| :--- | :--- | :--- |
| `id` | `int` | Unique identifier |
| `name` | `str` | User name |
