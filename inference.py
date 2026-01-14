from transformers import pipeline, logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

logging.set_verbosity(logging.CRITICAL)

model_path = "llama-2-7b-chat-guanaco"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=200,
)

prompt = "Who is Leonardo Da Vinci?"
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]["generated_text"])
