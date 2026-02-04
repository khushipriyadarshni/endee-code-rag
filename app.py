import streamlit as st
from dotenv import load_dotenv

import ingest
import query

load_dotenv()

st.set_page_config(page_title="Endee Code RAG", layout="wide")

st.title("Endee Code RAG — Repo-aware Q&A")

with st.sidebar:
    st.markdown("---")
    st.write("Uses local embeddings + Endee vector database (no OpenAI required).")

repo_url = st.text_input("GitHub repository URL", value="")
if st.button("Index Repo"):
    if not repo_url:
        st.error("Please provide a repository URL")
    else:
        with st.spinner("Cloning and indexing repository (this may take a while)..."):
            try:
                n = ingest.index_repo(repo_url)
                st.success(f"Indexed {n} chunks")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

st.markdown("---")

question = st.text_input("Ask a question about the indexed code:")
if st.button("Ask"):
    if not question:
        st.error("Please enter a question")
    else:
        with st.spinner("Searching and generating answer..."):
            try:
                ans = query.answer_question(question)
                st.markdown("**Answer:**")
                st.write(ans)
            except Exception as e:
                st.error(f"Query failed: {e}")
