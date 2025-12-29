from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import os
import torch

load_dotenv()
os.environ['HF_HOME'] = 'D:/huggingface_cache'

model_id = "meta-llama/Llama-3.2-1B"

print(f"Loading model: {model_id}")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=30,
    do_sample=False,
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt = "What is AI?"

print("Running inference...")
result = llm.invoke(prompt)
print("Result:", result)