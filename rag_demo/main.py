import sys
from .loader import load_and_chunk
from .index import VectorIndex
from .generator import answer

INDEX_DIR = ".rag_index"

def build(pdf_dir: str):
    print(f"Loading PDFs from {pdf_dir} ...")
    records = load_and_chunk(pdf_dir)
    print(f"Built {len(records)} chunks. Embedding ...")
    idx = VectorIndex.build(records)
    idx.save(INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}/")

def ask(question: str, top_k: int = 4):
    idx = VectorIndex.load(INDEX_DIR)
    chunks = idx.search(question, top_k=top_k)
    print("\nRetrieved chunks:")
    for c in chunks:
        print(f"  - {c['source']} chunk {c['chunk_id']} (score {c['score']:.3f})")
    print("\nAnswer:\n")
    print(answer(question, chunks))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Use: python -m rag_demo.main build <pdf_dir>")
        print("       python -m rag_demo.main ask \"your question\"")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2])
    elif cmd == "ask":
        ask(sys.argv[2])
