from pathlib import Path
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()

IGNORE_DIRS = {".git", "node_modules", "dist", "build", "__pycache__", "vendor"}
ALLOWED_EXT = {".js", ".ts", ".hbs", ".css", ".scss", ".html"}  # Ember / JS relevant


def iter_source_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        if path.suffix in ALLOWED_EXT:
            yield path


def chunk_file(path: Path, max_lines: int = 120, overlap: int = 20):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Konnte {path} nicht lesen: {e}")
        return []

    lines = text.splitlines()
    chunks = []
    start = 0
    n = len(lines)

    while start < n:
        end = min(start + max_lines, n)
        snippet = "\n".join(lines[start:end])
        chunks.append(
            {
                "file": str(path),
                "start_line": start + 1,
                "end_line": end,
                "code": snippet,
            }
        )
        if end == n:
            break
        start = end - overlap

    return chunks


def main():
    repo_path = Path(os.getenv("REPO_PATH", "./kursausschreibung")).resolve()
    index_path = Path(os.getenv("INDEX_PATH", "./repo_index.pkl")).resolve()

    if not repo_path.exists():
        raise SystemExit(f"Repo-Pfad existiert nicht: {repo_path}")

    print(f"Indexiere Repo: {repo_path}")
    all_chunks = []
    for file_path in iter_source_files(repo_path):
        file_chunks = chunk_file(file_path)
        all_chunks.extend(file_chunks)

    if not all_chunks:
        raise SystemExit("Keine Code-Chunks gefunden, prüfe Dateiendungen / Pfad.")

    texts = [c["code"] for c in all_chunks]
    print(f"{len(all_chunks)} Code-Snippets gefunden. Baue TF-IDF-Vektoren...")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)

    data = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "chunks": all_chunks,
    }

    joblib.dump(data, index_path)
    print(f"Fertig. Index gespeichert in: {index_path}")


if __name__ == "__main__":
    main()
