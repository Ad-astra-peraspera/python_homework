# Author: moqiHe
# Description: 
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class RAGGenerator:
    def __init__(self, model_name="Qwen/Qwen1.5-0.5B-Chat"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()

    def generate(self, query, contexts, max_length=512):
        prompt = f"你是一个环卫维修领域的专家，请根据以下资料回答问题：\n\n"
        for i, context in enumerate(contexts):
            prompt += f"资料{i+1}：{context}\n"
        prompt += f"\n问题：{query}\n回答："

        inputs = self.tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_length)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)


