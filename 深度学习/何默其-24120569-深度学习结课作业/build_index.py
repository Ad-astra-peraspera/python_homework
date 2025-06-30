# Author: moqiHe
# Description: 
from sentence_transformers import SentenceTransformer
import faiss
import json
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')

with open('diagnosis_rule_annotated.jsonl', 'r', encoding='utf-8') as f:
    docs = [json.loads(line) for line in f if line.strip()]

texts = [item['text'] for item in docs]
embeddings = model.encode(texts, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, 'index.faiss')
with open('texts.json', 'w', encoding='utf-8') as f:
    json.dump(texts, f, ensure_ascii=False, indent=2)

print("[✓] Index and texts saved.")
