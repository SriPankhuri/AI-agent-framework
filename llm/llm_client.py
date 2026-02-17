from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class HFLocalLLM:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        print("Loading local model... (first time takes 2–5 minutes)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # This is required by your controller
    def synthesize(self, user_input, results):
        context = f"User request: {user_input}\nResults: {results}\nWrite a final helpful report."
        return self.generate(context)
