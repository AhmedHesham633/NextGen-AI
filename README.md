# AI Clinical Decision Support Lite — RAG Starter Kit

A minimal, beginner-friendly RAG (Retrieval-Augmented Generation) scaffold for the **AI Clinical Decision Support Lite Hackathon**.

This project implements an end-to-end clinical evidence question-answering pipeline:
1. **Ingestion**: Parses medical guideline PDFs from `data/` and splits them into text chunks.
2. **Embeddings & Vector Database**: Computes embeddings locally using FastEmbed (`BAAI/bge-small-en-v1.5`) and stores them in ChromaDB (100% free, runs offline, no API key needed).
3. **Retrieval**: Searches the vector database for the most relevant guideline passages.
4. **Generation**: Uses Google Gemini (`gemini-2.5-flash-lite` via the official `google-genai` SDK) to produce grounded, structured clinical answers with exact citations and safe refusals.

---

## Architecture Overview

```
PDF Document (data/)
  └── PDF Parsing (PyPDFLoader)
      └── Chunking (RecursiveCharacterTextSplitter)
          └── Citation Metadata (document_name, page_number, chunk_id)
              └── Embeddings (Local FastEmbed / BAAI/bge-small-en-v1.5)
                  └── Vector Storage (ChromaDB)
                      └── Retrieval (similarity search)
                          └── Context Construction
                              └── Generation (Google Gemini / gemini-2.5-flash-lite)
                                  └── Grounded Output (JSON Schema with Citations & Refusal)
```

---

## Project Structure

| File / Folder | Purpose |
|---|---|
| `config.py` | Central configuration (paths, chunk size, top-k, Gemini model settings) |
| `ingest.py` | Loads PDFs from `data/`, chunks them, embeds locally, and builds ChromaDB |
| `query.py` | Retrieval interface — queries ChromaDB and displays relevant passages (No API key needed) |
| `generate.py` | Grounded generation module using Google Gemini (`google-genai`) |
| `pipeline.py` | End-to-end CLI pipeline (Retrieval + Gemini Grounded Generation) |
| `requirements.txt` | Python package dependencies |
| `.env.example` | Template for environment variables (`GEMINI_API_KEY`, `GEMINI_MODEL`) |
| `schema/response_schema.json` | JSON Schema enforcing structured output shape |
| `data/` | Contains the official `WHO_Hypertension_Guideline_2021.pdf` |
| `eval/` | Benchmark test cases for retrieval quality and refusal testing |

---

## Quick Setup Guide (Windows Command Prompt)

Follow these exact steps to set up and run the project using Windows Command Prompt (`cmd.exe`).

### Step 1 — Open CMD inside the project folder
Open Command Prompt and navigate to your project directory:
```cmd
cd path\to\RAG_Project
```

### Step 2 — Create the virtual environment
Create a virtual environment named `ragv`:
```cmd
python -m venv ragv
```

### Step 3 — Activate the virtual environment
Activate the `ragv` environment in Windows CMD:
```cmd
ragv\Scripts\activate
```
> **Tip:** After activation, your command prompt line will start with `(ragv)`, indicating the environment is active:
> ```cmd
> (ragv) C:\Users\...\RAG_Project>
> ```

### Step 4 — Upgrade pip
```cmd
python -m pip install --upgrade pip
```

### Step 5 — Install dependencies
Install all required libraries:
```cmd
pip install -r requirements.txt
```

### Step 6 — Create your `.env` file
Copy the `.env.example` template to create your `.env` configuration file:
```cmd
copy .env.example .env
```

### Step 7 — Configure your `.env` file
Open the newly created `.env` file in Notepad or VS Code. It should contain:
```env
EMBEDDING_PROVIDER=local
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
GEMINI_MODEL=gemini-2.5-flash-lite
```

> **Important notes about `.env`:**
> - Replace `YOUR_GEMINI_API_KEY_HERE` with your actual Google Gemini API key.
> - `GEMINI_API_KEY` is **only required for the generation stage** (`pipeline.py`).
> - Ingestion (`ingest.py`) and retrieval (`query.py`) work 100% free and locally **without needing any Gemini API key**.
> - **Never share or commit `.env`** to GitHub or version control (it is already in `.gitignore`).

---

## Running the Project

### Step 8 — Build the vector database
Index the bundled WHO hypertension guideline into ChromaDB:
```cmd
python ingest.py
```

**Expected successful output:**
```
=== Day 1 Starter: Ingestion Pipeline ===

Loading WHO_Hypertension_Guideline_2021.pdf ...
  -> 13 pages loaded

Created <N> chunks from 13 pages.

Embedding <N> chunks using 'local' provider ...
Done. Index saved to C:\...\RAG_Project\chroma_db/

Next step: run  python query.py "your question here"  to test retrieval.
```
> **Note:** `<N>` represents the total number of text chunks created (e.g. ~28 chunks). This number will change if you modify `CHUNK_SIZE` or `CHUNK_OVERLAP` in `config.py`.

---

### Step 9 — Test retrieval (No API key needed)
Search the vector database for top matching guideline chunks:
```cmd
python query.py "What is the target blood pressure for a patient with known cardiovascular disease?"
```

**What you will see in the output:**
- **Top retrieved chunks**: Rank order of the best matching passages.
- **Score**: Similarity relevance score (e.g. `0.761`).
- **Document & Page**: Source file name and exact page number (`page 9`).
- **Chunk ID**: Unique identifier for citation tracking (`WHO_Hypertension_Guideline_2021-p9-c...`).
- **Preview text**: Direct excerpt of the guideline text.

---

### Step 10 — Run the full end-to-end RAG pipeline
Retrieve evidence and generate a grounded, structured answer with Gemini:
```cmd
python pipeline.py "What is the target blood pressure for a patient with known cardiovascular disease?"
```

**What happens:**
1. **Retrieval**: ChromaDB fetches relevant guideline chunks.
2. **Context Construction**: Formats the retrieved passages with document and page metadata.
3. **Grounded Generation**: Gemini generates a direct recommendation, quotes supporting evidence, and provides citations strictly from the retrieved text.

---

### Step 11 — Test safe refusal (Out-of-scope question)
Test how the pipeline handles questions not covered by the guideline (e.g. breast cancer screening):
```cmd
python pipeline.py "What is the recommended breast cancer screening interval?"
```

**Refusal output:**
The system returns `confidence: "insufficient"`, refuses to guess or invent medical advice, and leaves citations empty:
```json
{
  "recommendation": "I cannot answer this question because the provided guideline context covers hypertension treatment and contains no information about breast cancer screening.",
  "evidence": "",
  "citations": [],
  "confidence": "insufficient"
}
```

---

## How to Run the Project Next Time

In future sessions, you do **NOT** need to recreate `ragv` or reinstall packages.

Simply open CMD, navigate to the folder, and activate the virtual environment:
```cmd
cd path\to\RAG_Project
ragv\Scripts\activate
```

Then run whichever script you need:
```cmd
python query.py "your question"
python pipeline.py "your question"
```

When you are done working, deactivate the environment:
```cmd
deactivate
```

---

## Troubleshooting

- **`Vector database not found`**: Run `python ingest.py` first to generate the ChromaDB index.
- **`GEMINI_API_KEY is missing`**: Ingestion and `query.py` run without a key. To use `pipeline.py`, add your API key to `.env`.
- **Rebuilding the index from scratch**: Delete the `chroma_db/` folder and re-run `python ingest.py`.
