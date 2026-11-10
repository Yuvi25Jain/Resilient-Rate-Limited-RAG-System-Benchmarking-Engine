import os
import re
from typing import List, Dict, Any
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

# arXiv Research Papers Config
PAPERS_CONFIG = [
    {
        "file_name": "Attention_Is_All_You_Need.pdf",
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762v7"
    },
    {
        "file_name": "Moshi.pdf",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "arxiv_id": "2410.00037v1"
    },
    {
        "file_name": "Dynamic_Chunking.pdf",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "arxiv_id": "2507.07955v2"
    }
]

def extract_chunks_from_pdf(pdf_path: str, title: str, arxiv_id: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """Extracts text page by page with structural section and page metadata."""
    reader = PdfReader(pdf_path)
    chunks_data = []
    
    current_section = "Main Content"
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        lines = text.split("\n")
        words = []
        
        for line in lines:
            # Match section headers e.g. "3.2 Attention", "2 Model Architecture"
            header_match = re.match(r"^(\d+(\.\d+)*\s+[A-Z][a-zA-Z\s]{2,30})", line.strip())
            if header_match:
                current_section = header_match.group(1).strip()
            words.extend(line.split())
        
        # Sliding Window Chunking
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            if len(chunk_words) < 30:
                continue
            
            chunk_text = " ".join(chunk_words)
            chunks_data.append({
                "text": chunk_text,
                "metadata": {
                    "paper": title,
                    "arxiv_id": arxiv_id,
                    "section": current_section,
                    "page": page_num
                }
            })
            
    return chunks_data

def build_vector_store():
    print("Initializing ChromaDB & SentenceTransformer Model...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    if "research_papers" in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection("research_papers")
        
    collection = chroma_client.create_collection(name="research_papers")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    all_texts = []
    all_metadatas = []
    all_ids = []
    
    global_id = 0
    papers_dir = "./papers"

    for config in PAPERS_CONFIG:
        pdf_path = os.path.join(papers_dir, config["file_name"])
        if not os.path.exists(pdf_path):
            print(f"Warning: Missing {config['file_name']} in ./papers directory! Check file names.")
            continue

        print(f"Processing: {config['title']}...")
        chunks = extract_chunks_from_pdf(pdf_path, config["title"], config["arxiv_id"])
        
        for c in chunks:
            all_texts.append(c["text"])
            all_metadatas.append(c["metadata"])
            all_ids.append(f"chunk_{global_id}")
            global_id += 1

    if all_texts:
        print(f"Encoding {len(all_texts)} total document chunks into ChromaDB...")
        embeddings = embedder.encode(all_texts, show_progress_bar=True).tolist()
        
        collection.add(
            ids=all_ids,
            embeddings=embeddings,
            documents=all_texts,
            metadatas=all_metadatas
        )
        print("\nSuccess! Vector database stored in folder ./chroma_db!")

if __name__ == "__main__":
    build_vector_store()