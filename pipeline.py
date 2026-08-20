"""
End-to-End Grounded RAG Pipeline CLI
------------------------------------
Combines local retrieval (ChromaDB) and grounded generation (Gemini)
into a clean, educational end-to-end clinical decision support tool.

Usage:
    python pipeline.py "What blood pressure threshold should trigger starting medication?"
"""
import json
import sys

import config
from query import load_index, retrieve, print_results
from generate import generate_grounded_answer


def run_pipeline(question: str):
    """Executes the complete RAG pipeline: retrieval followed by grounded generation."""
    print("=" * 65)
    print(" === AI Clinical Decision Support Lite — End-to-End RAG Pipeline ===")
    print("=" * 65)
    print(f"\nQuestion: {question}\n")

    # Step 1: Retrieval (Local Embeddings + ChromaDB)
    print("--- Step 1: Document Retrieval (ChromaDB) ---")
    try:
        vectordb = load_index()
        results = retrieve(vectordb, question)
        print_results(results)
    except Exception as e:
        print(f"\n[Error] Retrieval failed: {e}")
        print("Make sure you have run 'python ingest.py' first to build the Chroma DB index.")
        sys.exit(1)

    # Step 2: Generation (Gemini)
    print("--- Step 2: Grounded Generation (Gemini) ---")
    response = generate_grounded_answer(question, results)

    # Step 3: Formatted Output
    print("\n" + "=" * 65)
    print(" === Final Clinical Response ===")
    print("=" * 65)
    print(f"\nRecommendation:\n  {response.get('recommendation')}\n")
    print(f"Confidence Level: {response.get('confidence', 'UNKNOWN').upper()}\n")

    evidence = response.get("evidence")
    if evidence:
        print(f"Evidence Excerpt:\n  \"{evidence}\"\n")

    citations = response.get("citations", [])
    if citations:
        print("Citations:")
        for i, cit in enumerate(citations, 1):
            doc = cit.get("document", "Unknown")
            sec = cit.get("section", "N/A")
            page = cit.get("page", "?")
            print(f"  [{i}] Document: {doc} | Page: {page} | Section: {sec}")
    else:
        print("Citations: None (Refusal / Insufficient Evidence)")

    print("\nStructured JSON Response:")
    print(json.dumps(response, indent=2))
    print("=" * 65)


def main():
    if len(sys.argv) < 2:
        print('Usage: python pipeline.py "your question here"')
        print('Example: python pipeline.py "What is the target blood pressure for a patient with known cardiovascular disease?"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    run_pipeline(question)


if __name__ == "__main__":
    main()
