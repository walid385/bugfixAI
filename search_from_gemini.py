# search_from_gemini.py
import sys
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

from gemini_client import GeminiClient, GeminiConfig, extract_ticket_text

load_dotenv()


def load_index():
    index_path = os.getenv("INDEX_PATH", "./repo_index.pkl")
    data = joblib.load(index_path)
    return data["vectorizer"], data["matrix"], data["chunks"]


def search_in_code(query: str, top_k: int = 5):
    vectorizer, matrix, chunks = load_index()
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_idx = sims.argsort()[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        c = chunks[idx]
        score = float(sims[idx])
        results.append(
            {
                "rank": rank,
                "score": score,
                "file": c["file"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "code": c["code"],
            }
        )
    return results


def main():
    if len(sys.argv) < 2:
        print("Verwendung: python search_from_gemini.py <item_id>")
        sys.exit(1)

    item_id = int(sys.argv[1])

    # 1) Ticket aus Gemini holen
    config = GeminiConfig.from_env()
    client = GeminiClient(config)
    item = client.get_item(item_id)
    ticket_text = extract_ticket_text(item)
    if not ticket_text:
        print("Ticket hat keinen Text? Prüfe die Feldnamen in extract_ticket_text().")
        sys.exit(1)

    print(f"=== Ticket {item_id} ===")
    print(ticket_text)
    print("\n=== Suche im Repo ===")

    # 2) Im Code nach ähnlichen Stellen suchen
    results = search_in_code(ticket_text, top_k=5)

    for r in results:
        print("\n" + "=" * 80)
        print(f"Treffer #{r['rank']}  Score: {r['score']:.3f}")
        print(f"Datei: {r['file']} (Zeilen {r['start_line']}–{r['end_line']})")
        print("-" * 80)
        code_lines = r["code"].splitlines()
        preview_lines = code_lines[:30]
        print("\n".join(preview_lines))
        if len(code_lines) > 30:
            print("... (gekürzt)")


if __name__ == "__main__":
    main()
