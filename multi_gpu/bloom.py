import torch
import time

model_name = "bigscience/bloom-3b"

text = "Hello my name is"
max_new_tokens = 50

def generate_text(model, tokenizer, input_text):
    encoded_input = tokenizer(input_text, return_tensors="pt")
    output_seq = model.generate(input_ids = encoded_input(=['input_ids',].cuda()))
    return tokenizer.decode(output_seq[0],skip_special_tokens=True)


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_8bit = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(name)
generate_from_model(model_8bit, tokenizer)
