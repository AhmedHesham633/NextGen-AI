"""
Retrieval Module & CLI
----------------------
Loads the Chroma index built by ingest.py, retrieves the top-k most
relevant chunks for a clinical question, and displays them with full citation
metadata (document name, page number, chunk id, score).

Works 100% locally with NO API keys required.

Usage:
    python query.py "What is the target blood pressure for a patient with known cardiovascular disease?"
"""
import sys
from pathlib import Path
from langchain_chroma import Chroma

import config
from ingest import get_embedding_function


def load_index():
    """Loads the persisted ChromaDB vector index."""
    if not config.CHROMA_DIR.exists():
        print(f"\n[Error] Vector database not found at {config.CHROMA_DIR}/")
        print("Please run 'python ingest.py' first to index your documents.\n")
        sys.exit(1)

    embedding_fn = get_embedding_function()
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=str(config.CHROMA_DIR),
    )


def translate_if_arabic(question: str) -> str:
    """If the question contains Arabic text, translates it to English for optimal vector database retrieval."""
    is_arabic = any('\u0600' <= char <= '\u06FF' for char in question)
    if not is_arabic:
        return question

    if not config.GEMINI_API_KEY:
        return question

    try:
        from google import genai
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=f"Translate the following medical question into clear, concise English for searching an evidence database. Output ONLY the translated question and nothing else:\n{question}"
        )
        translated = response.text.strip()
        if translated:
            print(f"[Query Processing] Arabic query translated for vector search: '{question}' -> '{translated}'")
            return translated
    except Exception as e:
        print(f"[Query Processing Warning] Could not translate query: {e}")

    return question


def reformulate_query_with_history(question: str, chat_history: list = None) -> str:
    """If chat_history is present, reformulates follow-up questions into a standalone English search query."""
    if not chat_history:
        return translate_if_arabic(question)

    if not config.GEMINI_API_KEY:
        return translate_if_arabic(question)

    try:
        from google import genai
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        
        # Format last 4 turns of history
        history_lines = []
        for msg in chat_history[-4:]:
            role = "USER" if msg.get("role") == "user" else "ASSISTANT"
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")
            
        history_str = "\n".join(history_lines)
        
        prompt = f"""Given the following medical conversation history and a follow-up user question, rephrase the follow-up question into a standalone, fully self-contained medical question in English for searching a database.
Do NOT answer the question. Only output the standalone English search question.

Conversation History:
{history_str}

Follow-up Question: {question}
Standalone English Question:"""

        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt
        )
        standalone = response.text.strip()
        if standalone:
            print(f"[Query Context Reformulation] '{question}' -> '{standalone}'")
            return standalone
    except Exception as e:
        print(f"[Query Reformulation Warning] Could not reformulate query: {e}")

    return translate_if_arabic(question)


def retrieve(vectordb, question: str, k: int = None, chat_history: list = None):
    """Retrieves top-k relevant chunks for a question, supporting conversation history and multilingual queries."""
    k = k or config.TOP_K
    search_query = reformulate_query_with_history(question, chat_history)
    results = vectordb.similarity_search_with_relevance_scores(search_query, k=k)

    # Fallback search if reformulated query yielded no results
    if not results and search_query != question:
        results = vectordb.similarity_search_with_relevance_scores(question, k=k)

    return results




def print_results(results):
    """Prints retrieved chunks with similarity scores and citation metadata."""
    if not results:
        print("\nNo matching chunks found in the index.\n")
        return

    print(f"\nTop {len(results)} retrieved chunks:\n")
    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        doc_name = meta.get("document_name", "Unknown")
        page = meta.get("page_number", "?")
        chunk_id = meta.get("chunk_id", "N/A")
        print(f"[{i}] score={score:.3f}  Document: {doc_name}, page {page}, chunk {chunk_id}")
        preview = doc.page_content.strip().replace("\n", " ")[:200]
        print(f'    "{preview}..."\n')


def main():
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        print('Example: python query.py "What is the target blood pressure for a patient with known cardiovascular disease?"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Question: {question}")

    vectordb = load_index()
    results = retrieve(vectordb, question)
    print_results(results)


if __name__ == "__main__":
    main()
