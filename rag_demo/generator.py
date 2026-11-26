import ollama

SYSTEM_PROMPT = """You are my document assistant.
Answer the user's question by using the provided context only!
If the context does not contain the answer, say so.
Cite the source chunks you used by their tags."""

def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"[Source: {c['source']}, chunk {c['chunk_id']}]\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

def answer(question: str, chunks: list[dict], model: str = "llama3.2") -> str:
    prompt = build_prompt(question, chunks)
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.0},
    )
    return response["message"]["content"]
