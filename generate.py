import torch
import torch.nn as nn
import pickle
from gpt import GPTLanguageModel

model_path = "./model_2m.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPTLanguageModel()
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

"""
with open ('input.txt','r',encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))

#Tokenizer -> Mapping tokens to integers (Token = 1 character in our vocab)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # str -> [int]
decode = lambda l: ''.join([itos[i] for i in l]) # [int] -> str


def generate_text(model, context, max_new_tokens):
    with torch.no_grad():
        context = context.to(device)
        generated = model.generate(context, max_new_tokens)
        return decode(generated[0].tolist())

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = generate_text(model, context, max_new_tokens=1000)
print(generated_text)
"""