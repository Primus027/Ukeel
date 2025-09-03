# Split text
import json
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')

# Read full text
with open("amendment.txt", "r", encoding="utf-8") as f:
    full_text = f.read()
# Split into sentences
sentences = nltk.sent_tokenize(full_text)

# Grouping into chunks for RAG
chunks = []
current_chunk = []
current_len = 0
for groups in sentences:
    current_chunk.append(groups)
    current_len += len(groups.split())
    if current_len > 400:  # Adjust chunk size as needed
        chunks.append(" ".join(current_chunk))
        current_chunk = []
        current_len = 0
if current_chunk:
    chunks.append(" ".join(current_chunk))

# Saving chunks to JSON file
data = []
for i, chunk in enumerate(chunks):
    data.append({
        "id": i, 
        "text": chunk,
        "source": "Public-Procurement-Rules-2008-English.pdf"
    })
with open("rules.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

