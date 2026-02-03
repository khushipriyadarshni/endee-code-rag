import os
from typing import List
import openai

from dotenv import load_dotenv
from endee_store import search_chunks

load_dotenv()


def build_prompt(question: str, retrieved: List[tuple]) -> str:
    parts = [
        "You are a helpful assistant that answers questions about code. Use the provided code snippets to answer the user's question. If the answer is not present in the snippets, say you don't know. Be concise.",
        "---\nRetrieved snippets:\n",
    ]
    for i, (text, meta, score) in enumerate(retrieved, start=1):
        fn = meta.get("filename") if isinstance(meta, dict) else None
        parts.append(f"[Snippet {i} | file: {fn} | score: {score:.4f}]\n{text}\n---\n")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n".join(parts)


def answer_question(question: str, top_k: int = 3) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to answer questions")
    openai.api_key = api_key
    
    hits = search_chunks(question, k=top_k)
    prompt = build_prompt(question, hits)

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for answering code questions using context."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    return resp["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    q = input("Question: ")
    print(answer_question(q))
