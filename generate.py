"""
Grounded Generation Module (Gemini)
-----------------------------------
Uses Google's official GenAI SDK (google-genai) with structured output
to generate grounded clinical answers based strictly on retrieved context.

Structured Output Schema:
{
  "recommendation": "...",
  "evidence": "...",
  "citations": [{"document": "...", "section": "...", "page": 1}],
  "confidence": "high | medium | low | insufficient"
}
"""
import json
import os
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

import config

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

try:
    import jsonschema
except ImportError:
    jsonschema = None


# --- Pydantic Schema for Gemini Structured Output ---

class CitationModel(BaseModel):
    document: str = Field(description="Exact document name from metadata")
    section: str = Field(default="N/A", description="Section title or number if available, else 'N/A'")
    page: int = Field(description="Page number as an integer")


class GroundedResponseModel(BaseModel):
    recommendation: str = Field(description="Direct clinical answer or refusal message")
    evidence: str = Field(default="", description="Exact supporting text excerpt, or empty string if insufficient")
    citations: List[CitationModel] = Field(default_factory=list, description="List of source citations, empty if insufficient")
    confidence: Literal["high", "medium", "low", "insufficient"] = Field(description="Confidence level")


# Path to project JSON schema for validation
SCHEMA_PATH = config.BASE_DIR / "schema" / "response_schema.json"


def load_response_schema() -> Optional[dict]:
    """Loads the JSON schema from schema/response_schema.json if available."""
    if SCHEMA_PATH.exists():
        try:
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def format_context_for_prompt(retrieved_results: list) -> str:
    """Formats retrieved chunks with metadata for inclusion in the prompt."""
    context_blocks = []
    for doc, score in retrieved_results:
        meta = doc.metadata
        doc_name = meta.get("document_name", "Unknown Document")
        page = meta.get("page_number", 1)
        section = meta.get("section", "N/A")
        chunk_id = meta.get("chunk_id", "N/A")

        header = f"[Source Document='{doc_name}', Page={page}, Section='{section}', ChunkID='{chunk_id}']"
        context_blocks.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n".join(context_blocks)


def create_refusal_response(reason: str) -> dict:
    """Creates a standard refusal dictionary adhering to response_schema.json."""
    return {
        "recommendation": reason,
        "evidence": "",
        "citations": [],
        "confidence": "insufficient"
    }


def validate_citations(response: dict, retrieved_results: list) -> bool:
    """Validates that every citation returned by the LLM corresponds to a real
    document name and page number present in the retrieved chunks."""
    # Refusals or responses with no citations don't require citation checks
    if response.get("confidence") == "insufficient":
        return True

    citations = response.get("citations", [])
    if not citations:
        return True

    # Build the set of valid (document_name, page_number) pairs from retrieval metadata
    valid_sources = set()
    for doc, _ in retrieved_results:
        meta = doc.metadata
        doc_name = meta.get("document_name")
        page_num = meta.get("page_number")
        if doc_name is not None and page_num is not None:
            valid_sources.add((str(doc_name), int(page_num)))

    # Verify every citation against the allowed set
    for cit in citations:
        cit_doc = str(cit.get("document", ""))
        try:
            cit_page = int(cit.get("page", 0))
        except (ValueError, TypeError):
            print(f"\n[Citation Validation Error] Invalid page number format: {cit.get('page')}")
            return False

        if (cit_doc, cit_page) not in valid_sources:
            print(f"\n[Citation Validation Error] Hallucinated citation rejected: Document='{cit_doc}', Page={cit_page}")
            return False

    return True


def generate_grounded_answer(question: str, retrieved_results: list, chat_history: list = None) -> dict:
    """Generates a grounded answer using Gemini constrained by structured output
    and validates both the schema and citations before returning.
    Supports chat_history for context-aware follow-up conversations.
    """
    # 1. Validate API Key
    api_key = config.GEMINI_API_KEY
    if not api_key:
        print("\n[Notice] GEMINI_API_KEY is not set.")
        print("To generate answers with Gemini:")
        print("  1. Add your key to .env: GEMINI_API_KEY=your_key_here")
        print("  2. Re-run: python pipeline.py \"your question\"\n")
        return create_refusal_response("GEMINI_API_KEY is missing. Please set GEMINI_API_KEY in your .env file to enable generation.")

    # 2. Check SDK installation
    if genai is None:
        print("\n[Error] google-genai package is not installed.")
        print("Run: pip install google-genai\n")
        return create_refusal_response("google-genai package is missing. Please run 'pip install google-genai'.")

    # 3. Check for empty retrieval results
    if not retrieved_results:
        return create_refusal_response("لا يمكنني الإجابة على هذا السؤال لأنه لم يتم العثور على أي معلومات سريرية متعلقة به في المستندات المتاحة.")

    # 3b. Filter by minimum score threshold to avoid low-quality chunks
    min_threshold = getattr(config, "CONFIDENCE_LOW_THRESHOLD", 0.38)
    filtered_results = [(doc, score) for doc, score in retrieved_results if score >= min_threshold]

    if not filtered_results:
        return create_refusal_response(
            "لا تتوفر أدلة كافية في المستندات المتاحة للإجابة على هذا السؤال بدرجة ثقة مقبولة. "
            "يرجى إعادة صياغة السؤال أو التحقق من المصادر المتاحة."
        )

    # 3c. Determine expected confidence based on top score
    top_score = max(score for _, score in filtered_results)
    high_thresh   = getattr(config, "CONFIDENCE_HIGH_THRESHOLD",   0.65)
    medium_thresh = getattr(config, "CONFIDENCE_MEDIUM_THRESHOLD", 0.50)

    if top_score >= high_thresh:
        score_hint = "high"
    elif top_score >= medium_thresh:
        score_hint = "medium"
    else:
        score_hint = "low"

    # Use filtered results for generation
    retrieved_results = filtered_results


    context_str = format_context_for_prompt(retrieved_results)

    # Format Chat History if present
    history_section = ""
    if chat_history:
        history_lines = []
        for msg in chat_history[-4:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")
        history_section = "\nPrevious Conversation Context:\n" + "\n".join(history_lines) + "\n"

    # 4. Construct grounded prompt with strict instructions (Arabic Output + Conversation Memory)
    prompt = f"""You are a clinical decision support AI assistant. Your task is to answer the user's clinical question based strictly and ONLY on the provided Context below.
{history_section}
CRITICAL GROUNDING RULES:
1. Answer ONLY using facts directly stated in the Context.
2. Do NOT use outside medical knowledge, general background knowledge, or personal opinion.
3. Do NOT invent, extrapolate, or hallucinate medical advice, document names, section names, or page numbers.
4. LANGUAGE REQUIREMENT: You MUST write the "recommendation" and "evidence" fields in clear, professional Arabic (العربية الطبية السليمة والدقيقة).
5. CONVERSATION CONTEXT: If the user question is a follow-up question referencing prior discussion (e.g., "what about him?", "what are the side effects of this?"), use the conversation history to understand what entity or topic is being referred to, but ensure every claim in your recommendation is strictly supported by the retrieved Context.
6. CONFIDENCE HINT: The retrieval system scored the top result as '{score_hint}' confidence based on semantic similarity. Use this as a guide when setting your confidence level.
7. If the provided Context does NOT contain sufficient evidence to answer the question:
   - Set "confidence" to "insufficient"
   - Provide a concise refusal statement in Arabic in "recommendation" explaining that the evidence in the source document is insufficient to answer the question.
   - Set "evidence" to "" (empty string)
   - Set "citations" to [] (empty array)
8. If sufficient evidence IS present:
   - Provide a direct answer in Arabic in "recommendation".
   - Quote or translate the supporting text accurately in Arabic in "evidence".
   - Include citations in "citations" using the exact "Document" name (in English as in metadata) and "Page" number from the source metadata. If section name is unavailable or 'N/A' in metadata, use "N/A".
   - Set "confidence" to "high", "medium", or "low" based on the confidence hint and evidence quality.

Context:
{context_str}

User Question: {question}"""



    # 5. Execute Gemini call using google-genai SDK with structured output
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=GroundedResponseModel,
            )
        )

        response_text = response.text.strip()
        result = json.loads(response_text)

        # 6. Validate against JSON schema — fail safely if invalid
        schema = load_response_schema()
        if schema and jsonschema:
            try:
                jsonschema.validate(instance=result, schema=schema)
            except jsonschema.ValidationError as ve:
                print(f"\n[Validation Error] LLM output failed schema validation: {ve.message}")
                return create_refusal_response("The model response did not meet the required JSON schema.")

        # 7. Programmatic Citation Validation — reject hallucinated citations
        if not validate_citations(result, retrieved_results):
            return create_refusal_response(
                "The generated response contained citation metadata that was not present in the retrieved evidence."
            )

        return result

    except json.JSONDecodeError:
        print("[Error] Could not parse Gemini output as JSON.")
        return create_refusal_response("The model generated an invalid response format.")
    except Exception as e:
        error_msg = str(e)
        print(f"\n[Error] Gemini API Error: {error_msg}")
        if "API_KEY" in error_msg or "INVALID_ARGUMENT" in error_msg:
            print("Please check that your GEMINI_API_KEY in .env is valid.")
        elif "429" in error_msg or "QUOTA" in error_msg.upper():
            print("Gemini API rate limit / quota exceeded. Please try again later.")
        return create_refusal_response(f"Gemini API request failed: {error_msg}")


if __name__ == "__main__":
    print("generate.py is a module. Run pipeline.py to test the full end-to-end flow.")
