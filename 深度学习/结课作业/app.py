# Author: moqiHe
# Date: 2025-06-28
# Description: 
from fastapi import FastAPI
from pydantic import BaseModel
from rag_retriever import RAGRetriever
from rag_generator import RAGGenerator

app = FastAPI()
retriever = RAGRetriever()
generator = RAGGenerator()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def get_answer(req: QueryRequest):
    contexts = retriever.retrieve(req.query)
    answer = generator.generate(req.query, contexts)
    return {"question": req.query, "answer": answer, "contexts": contexts}

