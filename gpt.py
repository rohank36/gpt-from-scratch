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

#Hyperparameters
batch_size = 4
block_size = 8
emb_dim = 32
n_head = 16
vocab_size = len(chars)
max_iters = 1000
eval_interval = 100
eval_iters = 20
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_layers = 6
dropout = 0.2
#----------------
torch.manual_seed(1337)

#Tokenize Data
data = torch.tensor(encode(text), dtype=torch.long)

#Split data -> train/validation 
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

#Data Loading
def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #random batch size sample from data
    x = torch.stack([data[i:i+block_size] for i in ix]) #original
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #target
    x, y = x.to(device), y.to(device)
    return x,y

#Metrics during training
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ Single Head of Masked Self-Attention """
    def __init__(self, head_dim):
        super().__init__()
        self.WQ = nn.Linear(emb_dim, head_dim, bias=False) #query matrix
        self.WK = nn.Linear(emb_dim, head_dim, bias=False) #key matrix
        self.WV = nn.Linear(emb_dim, head_dim, bias=False) #value matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        query = self.WQ(inputs) #(4,8,16) = (batch_size, block_size, head_dim)
        key = self.WK(inputs) #(4,8,16) = (batch_size, block_size, head_dim)
        value = self.WV(inputs) #(4,8,16) = (batch_size, block_size, head_dim)
        attention_matrix = query @ key.transpose(-2,-1) / key.shape[-1]**-0.5 #(4,8,8) = (batch_size, block_size, block_size)
        #tril = torch.tril(torch.ones(block_size,block_size))
        attention_matrix = attention_matrix.masked_fill(self.tril[:block_size,:block_size] == 0, float('-inf'))
        attention_weights = F.softmax(attention_matrix, dim=-1) #softmax along the cols
        attention_weights = self.dropout(attention_weights)
        outputs = attention_weights @ value #(4,8,16) = (batch_size, block_size, head_dim)
        return outputs
    
class MultiHead(nn.Module):
    """ Multiple Heads of Masked Self-Attention in parallel """
    def __init__(self, n_head, head_dim, emb_dim):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_dim) for _ in range(n_head)])
        self.proj = nn.Linear(head_dim * n_head, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.heads], dim=-1) #(4,8,32) = (batch_size, block_size, emb_dim)
        output = self.dropout(self.proj(output)) #(4,8,32) = (batch_size, block_size, emb_dim)
        return output
    
class FeedForwardNetwork(nn.Module):
    """ Multi Layer Perceptron with ReLU """
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim,4*emb_dim),
            nn.ReLU(),
            nn.Linear(4*emb_dim, emb_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self,x):
        return self.net(x)

class DecoderBlock(nn.Module):
    """ Transformer Decoder Block """
    def __init__(self, emb_dim, n_head):
        super().__init__()
        head_dim = emb_dim // n_head
        self.self_attention = MultiHead(n_head, head_dim, emb_dim)
        self.feed_forward = FeedForwardNetwork(emb_dim)
        self.layer1_norm = nn.LayerNorm(emb_dim)
        self.layer2_norm = nn.LayerNorm(emb_dim)
    
    def forward(self, x):
        x = x + self.self_attention(self.layer1_norm(x)) # Pre Layer Norm implementation 
        x = x + self.feed_forward(self.layer2_norm(x))
        return x #(4,8,32) = (batch_size, block_size, emb_dim)

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb_table = nn.Embedding(block_size, emb_dim)
        self.blocks = nn.Sequential(*[DecoderBlock(emb_dim, n_head) for _ in range(n_layers)])
        self.layerFinal_norm = nn.LayerNorm(emb_dim) # Following OpenAIs GPT implementation
        self.final_linear = nn.Linear(emb_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs, targets=None):
        tok_emb = self.token_emb_table(inputs)
        pos_emb = self.pos_emb_table(torch.arange(block_size, device=device)) #(block_size, emb_dim)
        x = tok_emb + pos_emb #(batch_size, block_size, emb_dim)
        x = self.blocks(x) #(batch_size, block_size, emb_dim)
        x = self.layerFinal_norm(x) #(batch_size, block_size, emb_dim)
        logits = self.final_linear(x) #(batch_size, block_size, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C) #Reshaping to fit pytorch cross_entropy expected inputs shape
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, inputs, max_new_tokens):
        for _ in range (max_new_tokens):
            inputs_cond = inputs[:, -block_size:] #Context is the last block_size of tokens
            logits, loss = self(inputs_cond)
            logits = logits[:, -1, :] #(batch_size, emb_dim)
            probs = F.softmax(logits, dim=-1) #(batch_size, emb_dim)
            next = torch.multinomial(probs, num_samples=1) #Sample from prob distribution. (batch_size, 1)
            inputs = torch.cat((inputs, next), dim=1) #(batch_size, block_size + 1)
        return inputs

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for iter in range(max_iters):
    #Eval loss on train and valid sets every once in a while
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #Sample a batch of data
    xb, yb = get_batch('train')

    #Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

