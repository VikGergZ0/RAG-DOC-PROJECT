import re
from pathlib import Path
from pypdf import PdfReader


def load_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    text = re.sub(r"[\ue000-\uf8ff]", " ", text)   # strip private-use glyphs
    text = " ".join(text.split())

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def load_and_chunk(pdf_dir: str, chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    records = []
    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        text = load_pdf_text(pdf_path)
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            records.append({
                "text": chunk,
                "source": pdf_path.name,
                "chunk_id": i,
            })
    return records