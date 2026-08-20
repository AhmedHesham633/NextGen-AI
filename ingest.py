"""
Day 1 Starter — Ingestion Pipeline
-----------------------------------
Loads every PDF in ./data, splits it into overlapping chunks, embeds
those chunks, and stores them in a local ChromaDB collection. Every
chunk carries citation-ready metadata: document name, page number,
and a stable chunk id.

Usage:
    python ingest.py
"""
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import config



def get_embedding_function():
    """Returns the embedding function based on config.EMBEDDING_PROVIDER."""
    if config.EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL)
    else:
        from langchain_community.embeddings import FastEmbedEmbeddings
        return FastEmbedEmbeddings(model_name=config.LOCAL_EMBEDDING_MODEL)


def load_documents(data_dir: Path):
    """Loads PDFs, CSVs, and TXT files in data_dir and returns LangChain Documents,
    each carrying normalized metadata (document_name, page_number)."""
    pdf_files = sorted(data_dir.glob("*.pdf"))
    csv_files = sorted(data_dir.glob("*.csv"))
    txt_files = sorted(data_dir.glob("*.txt"))

    all_files = pdf_files + csv_files + txt_files
    if not all_files:
        print(f"No supported files (PDF, CSV, TXT) found in {data_dir}/")
        print("Add your files there, then re-run this script.")
        sys.exit(1)

    all_docs = []

    # Load PDFs
    for pdf_path in pdf_files:
        print(f"Loading {pdf_path.name} ...")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["document_name"] = pdf_path.stem
            page.metadata["page_number"] = page.metadata.get("page", 0) + 1
        all_docs.extend(pages)
        print(f"  -> {len(pages)} pages loaded")

    # Load CSVs
    max_rows = getattr(config, "MAX_CSV_ROWS", None)
    for csv_path in csv_files:
        print(f"Loading {csv_path.name} ...")
        try:
            loader = CSVLoader(str(csv_path), encoding="utf-8")
            csv_docs = loader.load()
        except Exception:
            loader = CSVLoader(str(csv_path), encoding="latin-1")
            csv_docs = loader.load()

        if max_rows and len(csv_docs) > max_rows:
            print(f"  -> Large CSV detected ({len(csv_docs)} rows). Sampling top {max_rows} rows for fast local embedding...")
            csv_docs = csv_docs[:max_rows]

        for i, doc in enumerate(csv_docs):
            doc.metadata["document_name"] = csv_path.stem
            doc.metadata["page_number"] = i + 1
        all_docs.extend(csv_docs)
        print(f"  -> {len(csv_docs)} rows loaded into index")


    # Load TXTs
    for txt_path in txt_files:
        print(f"Loading {txt_path.name} ...")
        loader = TextLoader(str(txt_path), encoding="utf-8")
        txt_docs = loader.load()
        for doc in txt_docs:
            doc.metadata["document_name"] = txt_path.stem
            doc.metadata["page_number"] = 1
        all_docs.extend(txt_docs)
        print(f"  -> {len(txt_docs)} document(s) loaded")

    return all_docs


def chunk_documents(documents):
    """Splits documents into overlapping chunks using a recursive splitter
    that prefers paragraph breaks, then sentence breaks, then words —
    a simple approximation of section-aware chunking."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * 4,       # ~4 chars per token estimate
        chunk_overlap=config.CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Attach a stable, citation-ready chunk_id to every chunk
    for i, chunk in enumerate(chunks):
        doc_name = chunk.metadata.get("document_name", "unknown")
        page = chunk.metadata.get("page_number", "?")
        chunk.metadata["chunk_id"] = f"{doc_name}-p{page}-c{i}"

    return chunks


def build_index(chunks):
    """Embeds chunks and persists them into a local Chroma collection."""
    embedding_fn = get_embedding_function()

    print(f"Embedding {len(chunks)} chunks using '{config.EMBEDDING_PROVIDER}' provider ...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
    )
    print(f"Done. Index saved to {config.CHROMA_DIR}/")
    return vectordb


def main():
    print("=== Day 1 Starter: Ingestion Pipeline ===\n")
    documents = load_documents(config.DATA_DIR)
    chunks = chunk_documents(documents)
    print(f"\nCreated {len(chunks)} chunks from {len(documents)} document pages/rows.\n")
    build_index(chunks)
    print('\nNext step: run  python query.py "your question here"  to test retrieval.')



if __name__ == "__main__":
    main()
