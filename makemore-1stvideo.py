import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

with open('names.txt', 'r') as f:
    data = f.read()
words = data.splitlines()
chars = sorted(list(set(str.join('', words))))


stoi = { s:(i+1) for i,s in enumerate(chars)}
itos = { (i+1): s for i,s in enumerate(chars)}
stoi['.'] = 0
itos[0] = '.'
N = torch.ones((27,27), dtype=torch.int32)
for w in words:
    N[stoi[w[0]], 0] += 1
    N[0,stoi[w[0]]] += 1
    for l in range(len(w) - 1):
        N[stoi[w[l]],stoi[w[l+1]]] += 1

g = torch.Generator().manual_seed(2147483647)

xs, ys = [],[]
# for word in words[:1]:
for word in words[:5]:
    tot = '.' + word + '.'
    ch1 = tot[:-1]
    ch2 = tot[1:]
    for t in range(len(ch1)):
        xs.append(stoi[ch1[t]])
        ys.append(stoi[ch2[t]])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
xenc = F.one_hot((xs), num_classes=27).float()
yenc = F.one_hot((ys), num_classes=27).float()
W = torch.randn((27, 27), generator=g, requires_grad=True)
regularization = 0.01
learningRate = 50
print(f'shapes are {xenc.shape}')
# nll (loss) that ys is output of xs * W
for k in range(100):
    # find product of input logits and weights, exp to make positive
    counts = (xenc @ W + ((W**2)*regularization).mean()).exp()
    # prob is softmax of counts
    prob = counts/counts.sum(1, keepdim=True)
    # prob[x,y] is the 
    # the loss is negative-log likelyhood the xend*W = yenc
    loss = (torch.log(prob[torch.arange(len(xs)), ys])*-1).mean()
    W.grad = None
    loss.backward()

    W.data += -1 * learningRate * W.grad
    print(f'guess is {loss} k is {k}')

for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        counts = (xenc @ W).exp()
        counts = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(counts, 1, replacement=True).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(f'{''.join(out)}')
                     
# plt.imshow(W.grad)
#print(counts.shape)
# plt.show()



