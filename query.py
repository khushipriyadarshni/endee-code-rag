from endee_store import search_chunks

def answer_question(question: str, top_k: int = 3) -> str:
    hits = search_chunks(question, k=top_k)

    if not hits:
        return "No relevant code found for this question."

    answer = "Answer based on retrieved code snippets:\n\n"

    for i, (text, meta, score) in enumerate(hits, start=1):
        filename = meta.get("filename") if isinstance(meta, dict) else "unknown"
        answer += f"[Snippet {i} | file: {filename} | score: {score:.4f}]\n"
        answer += text + "\n\n"

    return answer


if __name__ == "__main__":
    q = input("Question: ")
    print(answer_question(q))
