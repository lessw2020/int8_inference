

import torch
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 128
model_name = 'facebook/opt-66b'

text = """
Q: On average Joe throws 25 punches per minute. A fight lasts 5 rounds of 3 minutes. 
How many punches did he throw?\n
A: Let’s think step by step.\n"""


free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
max_memory = f'{free_in_GB-2}GB'

n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}

model = AutoModelForCausalLM.from_pretrained(
  model_name, 
  device_map='auto', 
  load_in_8bit=True, 
  max_memory=max_memory
)

# inference

tokenizer = AutoTokenizer.from_pretrained(model_name)
start = time.perfcounter()
input_ids = tokenizer(text, return_tensors="pt").input_ids

generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
end = time.perfcounter()
print(f"total inference time = {end - start} seconds")
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
