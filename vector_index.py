# Searchable knowledge base using embeddings and FAISS
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Loading the JSON file
with open("rules.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Creating embeddings
texts = [item["text"] for item in data]
embeddings = model.encode(texts, show_progress_bar=True)

# Converting to an array for FAISS
embeddings = np.array(embeddings).astype("float32")

# Creating the FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Saving the index
faiss.write_index(index, "real_rules.index")

with open("rules.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("FAISS index and JSON data saved successfully.")
