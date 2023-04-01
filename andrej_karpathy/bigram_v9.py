
import torch
import torch.nn as nn
from torch.nn import functional as F

##-- hyperparameters
batch_size = 64 ##-- сколько независимых объектов (предложений) обрабатывается за раз (параллельно)
block_size = 256 ##-- максимальная длина предложения (в нашем случае, количество символов)
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 ##-- размер эмбединга
n_head = 6 ##-- количество голов в multihead attention
n_layer = 6 ##-- количество блоков
dropout = 0.2
##------------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# here all the unique character that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
##-- create mapping from characters to integers
stoi = { ch: i for i, ch in enumerate(chars) }
atoi = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([atoi[i] for i in l])

##-- Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

##-- data loading
def get_batch(split: str='train'):
    '''
    generate a small batch of data of inputs x and targets y

    split: str: train or test
    '''
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, )) ##-- random idx
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return (x, y)

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
    ''' one head of self-attention '''

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # (B, T, C)
        k = self.key(x) # (B, T, C)
        ##-- compute attention score 'affinities' (взаимоотношения между токенами)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -->> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei) ##-- некоторые токены не будут взаимодействовать друг с другом
        # perform the weighted aggregation of the value
        v = self.value(x) # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -->> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    ''' multiple heads of self-attention in parallel '''

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) ##-- for residual connection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) ##-- объединение по каналам
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    ''' a simple linear layer followed by a non-linearity '''

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), ##-- for residual connection
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    ''' Transformer block: communication followed by computation '''

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) ##-- Layer Normalization by token (aka embedding)
        self.ln2 = nn.LayerNorm(n_embd) ##-- Layer Normalization by token (aka embedding)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) ##-- add residual connection ##-- and layer norm token
        x = x + self.ffwd(self.ln2(x)) ##-- add residual connection  ##-- and layer norm token
        return x


##-- super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        ##-- each token directly read oof the logits for the next token from a lookip table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) ##-- берем эмбединг токена (кодируем токен)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) ##-- кодируем позицию токена
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) ##-- MHA and FFWN n_layer times
        self.ln_f = nn.LayerNorm(n_embd) ##-- final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) ##-- linear_model head

    def forward(self, idx, targets=None):
        B, T = idx.shape

        ##-- idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C) #-- (4, 8, 32), (batch_size, block_size, emb_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C) (8, 32) ##-- кодируем позицию токена
        x = tok_emb + pos_emb ##-- (B, T, C) эмбединг токена и его позиции
        x = self.blocks(x) ##-- apply 4 heads of self-attention (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size) #-- (4, 8, 65), (batch_size, block_size, vocab_size)
        # print(logits.shape) # (4, 8, 65)
        # print(targets.shape) # (4, 8)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            ##-- из батча размером 4 предложения переводим в одну строку (то есть четыре предложения в одно)
            logits = logits.view(B*T, C)
            ##-- второй вариант, таргеты в таком случае тоже не менять
            # logits = torch.permute(logits, (0, 2, 1))
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens: int=100):
        ##-- idx is the (B, T) array of the indices in the current context
        for _ in range(max_new_tokens):
            ##-- crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            ##-- get the predictions
            logits, _ = self(idx_cond)
            ##-- focus only on the last time step (последнюю букву)
            # print(f"logits_before ={logits}, {logits.shape}")
            logits = logits[:, -1, :] # becomes (B, C)
            # print(f"logits_after ={logits}, {logits.shape}")
            ##-- softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # print(probs)
            ##-- sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # print(idx_next)
            ##-- append sampled idx to the running sample
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

##-- pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

##-- fit

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    ##-- sample a batch of data
    xb, yb = get_batch('train')

    ##-- evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

##-- generate the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

