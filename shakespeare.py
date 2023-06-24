import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
import argparse

from time import perf_counter

parser  = argparse.ArgumentParser(description = 'Model specifications')

parser.add_argument("-b",  "--batch_size", help='number of training examples in each batch', metavar='batch_size')
parser.add_argument("-t", "--block_size", help='Length of the time sequence')
parser.add_argument("-c", "--n_embd", help='dimension of the embedding dimension, also is the number of neuron in layers')
parser.add_argument("-nh", "--n_heads", help='number of heads for each multi-head attention layer')
parser.add_argument("-m", "--max_tokens", help='specify how many new tokens you want generated')
parser.add_argument("-s", "--save", action = "store_true", help='save the model after training')
parser.add_argument("-l", "--load", help='load the pretrained model off of the filepath')

args = parser.parse_args()


#GLOBALS
batch_size = args.batch_size if args.batch_size else 64 #This is the value of B
block_size = args.block_size if args.block_size else 256 #This is the value of T
n_embd = args.n_embd if args.n_embd else 512 #This is the value of C
n_head = args.n_heads if args.n_heads else 8
max_new_tokens = args.max_tokens if args.max_tokens else 1000

dropout = 0.2
n_layers = 8 #number of decoder blocks we will initialize
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 2e-4
ls = 0.0

eval_interval = 500
eval_iters = 200
max_iters = 5000


print(f"device is: {device}")

torch.manual_seed(1337)

#DATA PREPROCESSING
with open(os.getcwd()+'\\data\\input.txt', 'r', encoding='utf8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)


#character level mapping

s_to_i = {ch:i for i,ch in enumerate(chars)}
i_to_s = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [s_to_i[c] for c in s]
decode = lambda l: ''.join([i_to_s[c] for c in l])


data = torch.tensor(encode(text), dtype = torch.long)



n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#BATCH LOADER

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y, = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch('train')


#MODEL CLASSES

#Pytorch's positional encoding https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
            x: (T, B, C)
            We have to change our shape dimensions in to (T, B, C) and then change it back to (B, T, C) when done

        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class Head(nn.Module):
    """
    One Head of self-attention 
    """
    def __init__(self, head_size):
        super().__init__()

        self.head_size = head_size
        #initialize the key, query, and value matrices
        self.Wk = nn.Linear(n_embd, head_size, bias=False)
        self.Wq = nn.Linear(n_embd, head_size, bias=False)
        self.Wv = nn.Linear(n_embd, head_size, bias=False)

        #since this is a decoder, we need to initialize the mask as well

        #we register this as a buffer, which still exists as a 'matrix' to use, but we don't compute gradients on this or use in the backward pass
        #model parameters are objects that we use during the forward pass and we update using gradient descent
        #model buffers are objects that we use during computation but do not update

        #both parameters and buffers are saved to the right device when calling .to_device
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #assume x is of shape (B, T, C)
        B, T, C = x.shape
        K = self.Wk(x) #(B, T, head_size)
        Q = self.Wq(x) #(B, T, head_size)
        V = self.Wv(x) #(B, T, head_size)

        #K.T needs to be of shape (B, C, T), so we swap the -2 and -1 positions
        scores = Q @ K.transpose(-2, -1) * 1/(self.head_size)**(1/2) #(B, T, T)
        masked_scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B, T, T)
        attention_scores = F.softmax(masked_scores, dim = -1) #applying softmax along the rows (B, T, T)
        attention_scores = self.dropout(attention_scores) #(B, T, T)
        out = attention_scores @ V #(B, T, head_size)
        
        return out #(B, T, head_size)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)]) #(B, T, n_heads*head_size)
        self.proj = nn.Linear(n_heads * head_size, n_embd) #paper specifies a final linear layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #assume input x is of size (B, T, C)

        #Each Head returns a output of size (B, T, head_size), we concatenate along the final dimension so that our variable 'out' is now (B, T, n_heads*head_size)
        out = torch.cat([h(x) for h in self.heads], dim = -1) #(B, T, n_heads*head_size)
        out = self.proj(out) #(B, T, C)
        out = self.dropout(out) #(B, T, C)

        return out
    
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        #input and output will be of size (B, T, C)
        self.ff1 = nn.Linear(n_embd, 4*n_embd)
        self.ff2 = nn.Linear(4*n_embd, n_embd)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #assume x is of shape (B, T, C)
        x = self.ff1(x)
        x = self.relu(x)
        x = self.ff2(x)
        x = self.dropout(x)

        return x
    

class Block(nn.Module):
    #implementaion of one transformer block
    def __init__(self, n_head):
        super().__init__()
        self.head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, self.head_size)
        self.ffw = FeedForward()
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        #assume input of x is size (B, T, C) where x is the sum of the embedded input + positional_encoding
        x = self.layernorm1(x) #(B, T, C)
        x = x + self.sa(x) #(B, T, C)
        x = self.layernorm2(x) #(B, T, C)
        x = x + self.ffw(x) #(B, T, C)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embedding_matrix = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = PositionalEncoding(n_embd, dropout = dropout)
        # need '*' before list comprehension otherwise we get TypeError: list is not a Module subclass
        self.blocks = nn.Sequential(*[Block(n_head) for _ in range(n_layers)])

        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.final_linear = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, x, y = None):
        B, T = x.shape
        C = n_embd
        #assume our inputs x are of size (B, T)
        #assume our targets y are of size (B)

        token_embed = self.tok_embedding_matrix(x) #(B, T, C)

        pos_embed = self.pos_embedding(token_embed.view(T,B,C)).view(B, T, C) #(B, T, C)

        input = token_embed + pos_embed #(B, T, C)
        input = self.blocks(input) #(B, T, C)

        input = self.final_layer_norm(input) #(B, T, C)
        logits = self.final_linear(input) #(B, T, C)

        if y is not None:
            logits = logits.view(B*T, -1) #(B*T, C)
            y = y.view(B*T)
            loss = F.cross_entropy(logits, y, label_smoothing=ls)
        else:
            loss = None

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
    #generating some stuff
    #idx is (B, T) array of indices in our current context <-- current context of some list of characters in some batch
    #we keep extending (B, T) to (B, T+1), (B, T+2) and so on.. continuing until we reach max new tokens

        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            #get the predictions
            logits, loss = self(idx_cond) #<-- output of this is (B, T, C)
            #print(f"new dim of logits: {logits.shape}")
            #focus only on the last time step because the last time step is the prediction on what comes next
            logits = logits[:, -1, :] #becomes (B, C)
            #apply softmax to get the probabilities
            probs = F.softmax(logits, dim =-1) # (B, C)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)


            idx = torch.cat((idx, idx_next), dim = 1) #(B, T+1)


        return idx
#initializing the stuff

if not args.load:
    model = Decoder()
    m = model.to(device)

    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    #use this function to estimate the loss every once in a while
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

    #training loop
    print("Beginning training:")

    start_time = perf_counter()

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    stop_time = perf_counter()

    print("Elapsed time:", start_time, stop_time)

    if args.save:
        print("saving the model")
        filepath = os.getcwd()+"\\shakespeare_model\\model.pt"
        torch.save(model.state_dict(), filepath)
        print("model saved at:", filepath)

else:
    model = Decoder()
    model.load_state_dict(torch.load(args.load))
    model.eval()
    m = model.to(device)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=max_new_tokens)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))