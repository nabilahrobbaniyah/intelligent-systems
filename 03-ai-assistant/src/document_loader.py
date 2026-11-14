# AI ASSISTANT, DOCUMENT LOADER

import fitz

# Read PDF and extract text from each page
def load_pdf(pdf_path):
    documents = []
    pdf = fitz.open(pdf_path)

    for page_number, page in enumerate(pdf):
        text = page.get_text("text").strip()

        if text:
            documents.append({
                "text": text,
                "page": page_number + 1
            })

    pdf.close()
    return documents

# Split text into smaller chunks
def split_text(text, chunk_size=700, overlap=100):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap
    return chunks

# Convert extracted pages into chunks
def create_chunks(documents):
    all_chunks = []

    for document in documents:
        chunks = split_text(document["text"])

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "page": document["page"]
            })

    return all_chunks
