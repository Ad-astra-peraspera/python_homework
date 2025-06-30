# Author: moqiHe
# Description: 
import faiss
import json
from sentence_transformers import SentenceTransformer

class RAGRetriever:
    def __init__(self, index_path='index.faiss', texts_path='texts.json'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index(index_path)
        with open(texts_path, 'r', encoding='utf-8') as f:
            self.texts = json.load(f)

    def retrieve(self, query, top_k=3):
        query_vec = self.model.encode([query])
        scores, indices = self.index.search(query_vec, top_k)
        return [self.texts[i] for i in indices[0]]
