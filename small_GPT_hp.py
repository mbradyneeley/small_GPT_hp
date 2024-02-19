import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64
block_size = 256 # chars of context to predict the 257th
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # So every head has 64 dimensions as a standard
n_head = 6
n_layer = 6
dropout = 0.2
# -------------

torch.manual_seed(1337)

# read it to inspect it
with open('/mnt/atlas_local/brady/home/projects/karpathy_DL_tutorials/build_nanoGPT/small_GPT_hp/harry_potter_all_books.txt', 'r') as file:
    text = file.read()

# here are all the unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create mapping from characters to integers (simple tokenizer)
# Can map from string to int and back - for us we are doing this on a chatacter level. Many possible ways to do this.
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # starting index for each sequence, gets batch size amount of numbers
    x = torch.stack([data[i:i+block_size] for i in ix]) # batch of input sequences, these will be stacked as rows in a batch x block size tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # batch of target sequences
    x, y = x.to(device), y.to(device)
    return x, y

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
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        # typically people dont use biases here, I didnt either...
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # Softmax to make it a probability distribution
        wei = self.dropout(wei) # randomly prevent some of the nodes from communicating. Some nodes are randomly set to 0, ends up training an ensemble of subnetworks, finally everything is merged in the end. Can read paper for full detail.

        v = self.value(x) # v is the value of the token, what it emits
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    # Run heads in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    # Concatenate the heads
    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout), # can add right in the residual connection before coming back into the pathway
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer Block """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # communication followed by... vvv
        self.ffwd = FeedForward(n_embd) # computation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # departure from attention is all you need. We are using layer norm before it goes into self attention and liner layer
        x = x +self.sa(self.ln1(x))
        x = x +self.ffwd(self.ln2(x))
        return x

# Super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C) batch, time, channels
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C) time, channels
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, V) batch, time, vocab

        if targets is None:
            loss = None
        else:
            # evaluate loss function (Use neg log likelihood loss)
            # loss wants B, C, T order
            B, T, C = logits.shape

            logits = logits.view(B*T, C) # make 2d array
            targets = targets.view(B*T) # flatten
            loss = F.cross_entropy(logits, targets) # measure quality of logits with respect to targets; based on logits how well are we predicting the next character

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) tensor of integers or some context of characters from some batch
        for _ in range(max_new_tokens):
            # crop idx to last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append to the sequence
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# create the pytorch optimizer
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss() # averages loss over multiple batches
        print(f'step {iter}, train loss: {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    #eval the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
#print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
with open("harry_potter_text_nano_GPT_all_books.txt", "w") as file:
    generated_text = decode(m.generate(context, max_new_tokens=10000)[0].tolist())
    file.write(generated_text)
