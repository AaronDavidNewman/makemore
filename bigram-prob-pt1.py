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
# 27 27
# 27 1
psum = N.sum(dim=1, keepdim=True)
P = N/psum

print(psum.shape)
print (P.shape)
# plt.figure(figsize=(16,16))
# plt.rcParams.update({'font.size': 10})
# plt.imshow(N)
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha="center", va='bottom', color='gray')
#         plt.text(j, i, N[i, j].item(), ha="center", va='top', color='gray')

# plt.axis('off')

ix=0
words = []
for i in range(20):
    word = ''
    while True:
        p=P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        word = word + itos[ix]
        if ix == 0:
            print(word)
            words.append(word)
            break

xs, ys = [],[]
# for word in words[:1]:
# for word in words[:5]:
#     tot = '.' + word + '.'
#     ch1 = tot[:-1]
#     ch2 = tot[1:]
#     for t in range(len(ch1)):
#         xs.append(stoi[ch1[t]])
#         ys.append(stoi[ch2[t]])
#         prob = P[stoi[ch1[t]], stoi[ch2[t]]]
#         nlp = -1 * torch.log(prob)
#         print(f'{ch1[t]} {ch2[t]} {nlp:4f}')

# xs = torch.tensor(xs)
# ys = torch.tensor(ys)
# xenc = F.one_hot((xs), num_classes=27)
# yenc = F.one_hot((ys), num_classes=27)
# plt.imshow(xenc)
# plt.show()



