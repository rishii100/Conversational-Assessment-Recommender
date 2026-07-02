import numpy as np
from sentence_transformers import SentenceTransformer
from catalog import load_catalog
from config import EMBEDDING_MODEL
import os

def precompute():
    print(f"Loading model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Loading catalog...")
    catalog = load_catalog()
    texts = [item["search_text"] for item in catalog]
    
    print(f"Embedding {len(texts)} items...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    
    print("Saving to embeddings.npy...")
    np.save("embeddings.npy", embeddings)
    print("Done!")

if __name__ == "__main__":
    precompute()
