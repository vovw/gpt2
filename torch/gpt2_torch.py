import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

### hyper params
# model
ctx_len = 128
n_emb = 128
dropout = 0.1
head_size = 128
n_heads = 4 
n_layers = 3

# training
num_epochs = 20
batch_size = 64
lr = 1e-3

# device = torch.device("cuda" if torch.cuda.is_available() elif "mps" torch.backends.mps.is_available else "cpu")
device = torch.device("mps" if torch.cuda.is_available() else "cpu")

### Tokenization
with open('mahabharat.txt', 'r', encoding='utf-8') as f:
    text = f.read()
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
itos = {i:c for i,c in enumerate(vocab)} # int to string
stoi = {c:i for i,c in enumerate(vocab)} # string to int
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[i] for i in x])
data = encode(text)
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

### Data Prep
ctx_len = 8
X_train = torch.tensor([train_data[i:i+ctx_len] for i in range(0, len(train_data) - ctx_len, ctx_len)], dtype=torch.long)
y_train = torch.tensor([train_data[i+1:i+ctx_len+1] for i in range(0, len(train_data) - ctx_len, ctx_len)], dtype=torch.long)
X_val = torch.tensor([val_data[i:i+ctx_len] for i in range(0, len(val_data) - ctx_len, ctx_len)], dtype=torch.long)
y_val = torch.tensor([val_data[i+1:i+ctx_len+1] for i in range(0, len(val_data) - ctx_len, ctx_len)], dtype=torch.long)

def get_batches(X, y, b_size, shuffle=True):
    if shuffle:
        idx = torch.randperm(X.shape[0])
        X = X[idx]
        y = y[idx]
    for i in range(0, X.shape[0], b_size):
        yield X[i:i+b_size].to(device), y[i:i+b_size].to(device)

### Model Definition
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(n_emb, head_size, bias=False)
        self.q_proj = nn.Linear(n_emb, head_size, bias=False)
        self.v_proj = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer("_causal_mask", torch.triu(torch.ones(ctx_len, ctx_len) * float('-inf'), diagonal=1))
        self.c_proj = nn.Linear(head_size, n_emb)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        K = self.k_proj(x).view(B, T, n_heads, head_size // n_heads).transpose(1, 2)
        Q = self.q_proj(x).view(B, T, n_heads, head_size // n_heads).transpose(1, 2)
        V = self.v_proj(x).view(B, T, n_heads, head_size // n_heads).transpose(1, 2)
        
        attn_weights = (Q @ K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        attn_weights = attn_weights + self._causal_mask[:T, :T]
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        o = (attn_weights @ V).transpose(1, 2).contiguous().view(B, T, head_size)
        o = self.c_proj(self.resid_dropout(o))
        return o

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_emb, 4 * n_emb)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.mha = MultiHeadAttention()
        self.ln_1 = nn.LayerNorm(n_emb)
        self.ln_2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.mha(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_emb)
        self.wpe = nn.Embedding(ctx_len, n_emb)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)
        self._init_parameters()

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.wte(x)
        pos_emb = self.wpe(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, max_new_tokens):
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            logits = self(ctx[:, -ctx_len:])
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            ctx = torch.cat((ctx, next_tok), dim=1)
        return ctx

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'c_proj' in name and 'weight' in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
            elif 'weight' in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

### Training
model = GPT().to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    batch_cnt = 0
    for input, label in get_batches(X_train, y_train, batch_size):
        batch_cnt += 1
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits.view(-1, vocab_size), label.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / batch_cnt

    model.eval()
    running_loss = 0
    batch_cnt = 0
    with torch.no_grad():
        for input, label in get_batches(X_val, y_val, batch_size):
            batch_cnt += 1
            logits = model(input)
            loss = criterion(logits.view(-1, vocab_size), label.view(-1))
            running_loss += loss.item()
    avg_val_loss = running_loss / batch_cnt
    print(f"Epoch {epoch:2} | train = {avg_train_loss:.4f} | val = {avg_val_loss:.4f}")

### Inference
model.eval()
completion = decode(model.generate(1000)[0].cpu().tolist())
print(completion)
with open('completions.txt', 'w') as f:
    f.write(completion)
