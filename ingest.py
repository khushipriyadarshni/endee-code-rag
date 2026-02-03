import os
import shutil
from typing import List, Dict
from git import Repo, GitCommandError
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter

from endee_store import add_chunks



def clone_repo(url: str, dest: str = "repo") -> str:
    dest_path = Path(dest)
    if dest_path.exists():
        try:
            repo = Repo(dest_path)
            origin = repo.remotes.origin
            origin.fetch()
            origin.pull()
            return str(dest_path)
        except Exception:
            # remove and reclone
            shutil.rmtree(dest_path)
    try:
        Repo.clone_from(url, dest)
        return str(dest_path)
    except GitCommandError as e:
        raise RuntimeError(f"Failed to clone repository: {e}")


def _gather_files(root: str, exts=(".py", ".js", ".java")) -> List[Path]:
    p = Path(root)
    files = []
    for fp in p.rglob("*"):
        if fp.is_file() and fp.suffix in exts:
            files.append(fp)
    return files


def index_repo(repo_url: str, chunk_size: int = 500, overlap: int = 50):
    """Clone the repo, extract code files, chunk, and store into Endee."""
    repo_path = clone_repo(repo_url, dest="repo")
    files = _gather_files(repo_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    all_chunks = []
    metadatas = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        chunks = splitter.split_text(text)
        for c in chunks:
            all_chunks.append(c)
            metadatas.append({"filename": str(f.relative_to(repo_path))})

    # batch add (simple call)
    if all_chunks:
        add_chunks(all_chunks, metadatas)
    return len(all_chunks)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest.py <git-repo-url>")
        raise SystemExit(1)
    url = sys.argv[1]
    n = index_repo(url)
    print(f"Indexed {n} chunks")
