# AI ASSISTANT, RAG ENGINE

import ollama

def generate_answer(question, retrieved_chunks, model_name="llama3.2:3b"):

    if not retrieved_chunks:
        return "No relevant information was found."

    context = ""

    for i, chunk in enumerate(retrieved_chunks, 1):
        context += (
            f"\n[Source {i} | Page {chunk['page']}]\n"
            f"{chunk['text']}\n"
        )

    system_prompt = """
    You are an Industrial Technical Assistant.

    Answer using only the provided document context.

    Rules:
    1. Do not invent technical specifications.
    2. Do not fabricate maintenance procedures.
    3. If information is unavailable, say so.
    4. Cite source page numbers.
    5. For safety-critical questions, refer users
       to official manuals and qualified personnel.
    """

    user_prompt = f"""
    DOCUMENT CONTEXT:
    {context}

    USER QUESTION:
    {question}

    Answer based on the document and cite page numbers.
    """
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response["message"]["content"]