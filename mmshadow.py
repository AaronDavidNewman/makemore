import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

device=torch.device('cuda')
canCuda = torch.cuda.is_available()
print(f'canCuda is {canCuda} list is {torch.cuda.get_arch_list()}')
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
print(f'dtype is {dtype}')
with open('shakes_input.txt', 'r') as f:
    data = f.read()
words = data.splitlines()
chars = sorted(list(set(str.join('', words))))


stoi = { s:(i+1) for i,s in enumerate(chars)}
itos = { (i+1): s for i,s in enumerate(chars)}
stoi['.'] = 0
itos[0] = '.'
batch_size = 32
block_size=3
embeddingDim=5
hiddenDim = 300
# learningRate = 10** torch.linspace(-3, 0, 1000)
learningRate = 0.1

a=torch.tril(torch.ones(3,3))
asum = torch.sum(a, 1)
print(f'asum: {asum}')
X, Y = [], []
print(len(words))
for w in words:
    # print(w)
    # context = [0]*block_size # like torch.zeros(block_size)  
    context = torch.zeros(block_size)  
    for tt in w + '.':
        ix = stoi[tt]
        X.append(context)
        Y.append(ix)
        # print(''.join(itos[i.item()] for i in context), '==>', itos[ix])
        # context = torch.tensor(list(context[1:])+[ix])
        context = torch.cat((context[1:], torch.tensor([ix])))

g = torch.Generator().manual_seed(2147483647)
# X is embedding of input layer
X = torch.stack(X, dim=0).int()
Y = torch.tensor(Y)
vocabSize = len(words)+1 # +1 for the end token (.)
# C has weights of input layer
C = torch.randn((vocabSize, embeddingDim), generator=g)
# W1 weights of hidden layer
W1 = torch.randn((embeddingDim*block_size, hiddenDim), generator=g)
# b1 = torch.randn(hiddenDim, generator=g)
# W2 is weights of the output layer
W2 = torch.randn((hiddenDim, vocabSize), generator=g)
# b2 = torch.randn(vocabSize, generator=g)
print(f'X.shape is {X.shape}')
print(f'first 3 x is {X[0]}, {X[1]}, {X[2]}')
parameters = [C, W1, W2]
for p in parameters:
    p.requires_grad = True
# for i in range(15000)
for i in range(1000):
    ix = torch.randint(0, X.shape[0], (batch_size,), generator=g)
    emb = C[X[ix]]
    # h = torch.tanh(emb.reshape(-1, embeddingDim*block_size)@W1 + b1)
    h = torch.tanh(emb.reshape(-1, embeddingDim*block_size)@W1)
    # counts = h @ W2 + b2
    counts = h @ W2
    loss = F.cross_entropy(counts, Y[ix])
    if i > 8999:
        lr = learningRate * 0.1
    else:
        lr = learningRate
    if i % 200 == 0:
        print(f'{i}: {loss.item()}')
    for p in parameters:
        p.grad = None
    loss.backward()
    for p in parameters:
        p.data += -1 * lr*p.grad
    # probsm = counts/counts.sum(1, keepdim=True)
    # nll = (-1 * probsm[torch.arange(Y.shape[0]), Y].log()).mean()
    # print(nll)
print(loss.item())

