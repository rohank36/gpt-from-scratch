import torch
import torch.nn as nn 
from torch.nn import functional as F

#Load file
with open ('input.txt','r',encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

#Tokenizer -> Mapping tokens to integers (Token = 1 character in our vocab)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # str -> [int]
decode = lambda l: ''.join([itos[i] for i in l]) # [int] -> str

#Global Vars
batch_size = 4
block_size = 8
emb_dim = 32
n_head = 16
vocab_size = len(chars)

#Tokenize Data
data = torch.tensor(encode(text), dtype=torch.long)

#Split data -> train/validation 
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

#Get Batch
def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #random batch size sample from data
    x = torch.stack([data[i:i+block_size] for i in ix]) #original
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #target
    return x,y


#Single Head of Masked Self-Attention
