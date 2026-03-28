#!/usr/bin/env python3
"""Word2Vec (Skip-gram) — minimal zero-dep implementation."""
import random, math
from collections import Counter

def tokenize(text): return text.lower().split()

class Word2Vec:
    def __init__(self, dim=10, window=2, lr=0.025, epochs=50):
        self.dim=dim; self.window=window; self.lr=lr; self.epochs=epochs
    def fit(self, corpus):
        words=[w for s in corpus for w in tokenize(s)]
        vocab=list(set(words)); self.w2i={w:i for i,w in enumerate(vocab)}
        V=len(vocab); self.W=[[random.gauss(0,0.1) for _ in range(self.dim)] for _ in range(V)]
        self.C=[[random.gauss(0,0.1) for _ in range(self.dim)] for _ in range(V)]
        for _ in range(self.epochs):
            for i,w in enumerate(words):
                wi=self.w2i[w]
                start=max(0,i-self.window); end=min(len(words),i+self.window+1)
                for j in range(start,end):
                    if j==i: continue
                    ci=self.w2i[words[j]]
                    dot=sum(self.W[wi][d]*self.C[ci][d] for d in range(self.dim))
                    sig=1/(1+math.exp(-max(-10,min(10,dot))))
                    g=(1-sig)*self.lr
                    for d in range(self.dim):
                        self.W[wi][d]+=g*self.C[ci][d]
                        self.C[ci][d]+=g*self.W[wi][d]
    def similar(self, word, top=5):
        if word not in self.w2i: return []
        wi=self.w2i[word]; v=self.W[wi]
        nv=math.sqrt(sum(x**2 for x in v))
        sims=[]
        for w,i in self.w2i.items():
            if w==word: continue
            u=self.W[i]; nu=math.sqrt(sum(x**2 for x in u))
            if nv*nu==0: continue
            cos=sum(a*b for a,b in zip(v,u))/(nv*nu)
            sims.append((cos,w))
        sims.sort(reverse=True); return sims[:top]

if __name__=="__main__":
    corpus=["the king rules the kingdom","the queen rules the kingdom","the prince is young",
            "the princess is young","king and queen rule together","prince and princess play together"]
    random.seed(42); w2v=Word2Vec(dim=20,window=2,epochs=100); w2v.fit(corpus)
    for w in ["king","queen","prince"]:
        sims=w2v.similar(w,3)
        print(f"{w}: {[(s,f'{c:.2f}') for c,s in sims]}")
