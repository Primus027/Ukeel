#backend

import json
import os
import numpy as np
import faiss
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Load the JSON file
with open("rules.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Rebuilding FAISS index from embeddings
index = faiss.read_index("real_rules.index")

# Model for query embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, top_k=3):
    query_vec = embedder.encode([query]).astype("float32")
    D, I = index.search(query_vec, top_k)
    return [data[i] for i in I[0]]

# Gemini LLM Inference API setup
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
GEMINI_API_KEY = "AIzaSyBMPH9ZIQaiFI6jEW5D5Lkq_dl3Ejt6K_U"
def call_llm(context, question, timeout=30):
    # if not GEMINI_API_KEY:
    #     return {"error": "Gemini API key not found."}

    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        
        result = response.json()
        if "candidates" in result and len(result["candidates"]) > 0:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "Error: Unexpected API response format from Gemini."
            
    except requests.exceptions.Timeout:
        return f"Error: The request timed out after {timeout} seconds."
    # except requests.exceptions.RequestException as e:
    #     return f"Error communicating with the Gemini API: {e}"

# FastAPI setup
app = FastAPI()
class Query(BaseModel):
    query: str

@app.post("/ask")
def ask(query: Query):
    # Context retrieval
    chunks = retrieve(query.query)
    context = " ".join([c["text"] for c in chunks])
    # LLM call
    answer = call_llm(context, query.query)
    return {"answer": answer, "context": context}
    