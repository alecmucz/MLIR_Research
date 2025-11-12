import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load env vars
load_dotenv()

# Authenticate
login(token=os.environ["HF"])

# Load model
pipe = pipeline("text-generation", model="facebook/llm-compiler-7b")
tokenizer = AutoTokenizer.from_pretrained("facebook/llm-compiler-7b")
model = AutoModelForCausalLM.from_pretrained("facebook/llm-compiler-7b")

# Inference
def generate_code(prompt):
    return pipe(prompt, max_length=100)[0]['generated_text']
