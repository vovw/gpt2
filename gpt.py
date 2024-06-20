import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils
import numpy as np
import math

# Hyperparameters for the model and training
ctx_len = 128  # Context length for the model
n_emb = 128    # Embedding size
dropout = 0.1  # Dropout rate
head_size = 128  # Size of each attention head
n_heads = 4      # Number of attention heads
n_layers = 3     # Number of transformer blocks

num_epochs = 20  # Number of epochs to train
batch_size = 64  # Batch size for training
lr = 1e-3        # Learning rate

# Load and preprocess the text data
with open('./input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create a vocabulary from the text
vocab = sorted(set(text))
vocab_size = len(vocab)
itos = {i: c for i, c in enumerate(vocab)}  # Index to string mapping
stoi = {c: i for i, c in enumerate(vocab)}  # String to index mapping

# Encode the text into indices
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[i] for i in x])
data = encode(text)

# Split data into training and validation sets
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

# Prepare the data for training
ctx_len = 8  # Redefining context length for training purposes
X_train = mx.array([train_data[i:i+ctx_len] for i in range(0, len(train_data) - ctx_len, ctx_len)])
y_train = mx.array([train_data[i+1:i+ctx_len+1] for i in range(0, len(train_data) - ctx_len, ctx_len)])
X_val = mx.array([val_data[i:i+ctx_len] for i in range(0, len(val_data) - ctx_len, ctx_len)])
y_val = mx.array([val_data[i+1:i+ctx_len+1] for i in range(0, len(val_data) - ctx_len, ctx_len)])

# Function to generate batches of data
def get_batches(X, y, b_size, shuffle=True):
    if shuffle:
        ix = np.arange(X.shape[0])
        np.random.shuffle(ix)
        ix = mx.array(ix)
        X = X[ix]
        y = y[ix]
    for i in range(0, X.shape[0], b_size):
        yield X[i:i+b_size], y[i:i+b_size]

# Define the GPT model
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_emb)  # Token embeddings
        self.wpe = nn.Embedding(ctx_len, n_emb)     # Positional embeddings
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])  # Transformer blocks
        self.ln_f = nn.LayerNorm(dims=n_emb)  # Final layer normalization
        self.lm_head = nn.Linear(n_emb, vocab_size)  # Output layer
        self._init_parameters()  # Initialize parameters

    def __call__(self, x):
        B, T = x.shape
        tok_emb = self.wte(x)  # Token embeddings
        pos_emb = self.wpe(mx.arange(T))  # Positional embeddings
        x = tok_emb + pos_emb  # Combine embeddings
        x = self.blocks(x)  # Pass through transformer blocks
        x = self.ln_f(x)  # Apply final layer normalization
        logits = self.lm_head(x)  # Compute logits
        return logits

    def generate(self, max_new_tokens):
        ctx = mx.zeros((1, 1), dtype=mx.int32)  # Start with an empty context
        for _ in range(max_new_tokens):
            logits = self(ctx[:, -ctx_len:])
            logits = logits[:, -1, :]
            next_tok = mx.random.categorical(logits, num_samples=1)
            ctx = mx.concatenate((ctx, next_tok), axis=1)
        return ctx

    def _init_parameters(self):
        # Initialize parameters with specific distributions
        normal_init = nn.init.normal(mean=0.0, std=0.02)
        residual_init = nn.init.normal(mean=0.0, std=(0.02 / math.sqrt(2 * n_layers)))
        new_params = []
        for name, module in self.named_modules():
            if isinstance(module, nn.layers.linear.Linear):
                if 'c_proj' in name:
                    new_params.append((name + '.weight', residual_init(module.weight)))
                else:
                    new_params.append((name + '.weight', normal_init(module.weight)))
                if module.bias is not None:
                    new_params.append((name + '.bias', mx.zeros(module.bias.shape)))
            elif isinstance(module, nn.layers.embedding.Embedding):
                new_params.append((name + '.weight', normal_init(module.weight)))
        self.update(utils.tree_unflatten(new_params))

# Define additional components of the model (MultiHeadAttention, MLP, Block) as needed

# Define the training loop
model = GPT()
optimizer = optim.AdamW(learning_rate=lr)
for epoch in range(num_epochs):
    model.train(True)
    running_loss = 0
    batch_cnt = 0
    for input, label in get_batches(X_train, y_train, batch_size):
        batch_cnt += 1
        loss = loss_fn(model, input, label)
        optimizer.update(model.parameters(), loss.grad)
        running_loss += loss.item()
    avg_train_loss = running_loss / batch_cnt
    print(f"Epoch {epoch:2} | Train Loss: {avg_train_loss:.4f}")

    # Validation loop
    model.train(False)
    running_loss = 0
    batch_cnt = 0
    for input, label in get_batches(X_val, y_val, batch_size):
        batch_cnt += 1
        loss = loss_fn(model, input, label)
        running_loss += loss.item()
    avg_val_loss = running_loss / batch_cnt
    print(f"Epoch {epoch:2} | Validation Loss: {avg_val_loss:.4f}")

# Inference
completion = decode(model.generate(1000)[0].tolist())
print(completion)
with open('completions.txt', 'w') as f:
    f.write(completion)