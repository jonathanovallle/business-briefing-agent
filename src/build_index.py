import os, json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

DATA_DIR = "src/data/docs"
INDEX_PATH = "src/data/faiss_index.bin"
META_PATH = "src/data/faiss_meta.json"
EMBED_MODEL = "all-MiniLM-L6-v2"

def load_docs(data_dir=DATA_DIR):
    docs = {}
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        if os.path.isfile(path) and fname.lower().endswith(('.md','.txt','.text','.rst')):
            with open(path, "r", encoding="utf-8") as f:
                docs[fname] = f.read()
    return docs

def build_index():
    docs = load_docs()
    ids = []
    texts = []
    for i,(k,v) in enumerate(docs.items()):
        ids.append(k)
        texts.append(v)

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

    meta = {"ids": ids}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Index saved: {INDEX_PATH}, meta: {META_PATH}, docs indexed: {len(ids)}")

if __name__ == "__main__":
    build_index()