#!/usr/bin/env python3
"""word2vec_mini - Minimal Word2Vec skip-gram."""
import sys, random, math, collections, re
def tokenize(text): return re.findall(r"[a-z]+", text.lower())
class Word2Vec:
    def __init__(self, dim=10, window=2, neg_samples=5, lr=0.025):
        self.dim=dim; self.window=window; self.neg=neg_samples; self.lr=lr
    def train(self, text, epochs=5):
        words = tokenize(text); vocab = list(set(words)); self.w2i = {w:i for i,w in enumerate(vocab)}
        n = len(vocab); self.W = [[random.gauss(0,0.1) for _ in range(self.dim)] for _ in range(n)]
        self.C = [[random.gauss(0,0.1) for _ in range(self.dim)] for _ in range(n)]
        freq = collections.Counter(words)
        table = []; 
        for w, c in freq.items(): table.extend([self.w2i[w]] * int(c**0.75))
        for epoch in range(epochs):
            for i, w in enumerate(words):
                wi = self.w2i[w]
                for j in range(max(0,i-self.window), min(len(words), i+self.window+1)):
                    if i==j: continue
                    ci = self.w2i[words[j]]
                    dot = sum(self.W[wi][d]*self.C[ci][d] for d in range(self.dim))
                    sig = 1/(1+math.exp(-max(-10,min(10,dot))))
                    grad = self.lr*(1-sig)
                    for d in range(self.dim):
                        self.W[wi][d]+=grad*self.C[ci][d]; self.C[ci][d]+=grad*self.W[wi][d]
    def similar(self, word, top=5):
        if word not in self.w2i: return []
        wi = self.w2i[word]; vec = self.W[wi]
        sims = []
        for w, i in self.w2i.items():
            if w==word: continue
            dot = sum(vec[d]*self.W[i][d] for d in range(self.dim))
            na = math.sqrt(sum(v**2 for v in vec)); nb = math.sqrt(sum(v**2 for v in self.W[i]))
            sims.append((w, dot/(na*nb) if na and nb else 0))
        return sorted(sims, key=lambda x: -x[1])[:top]
if __name__=="__main__":
    text = "the king and queen lived in a castle the king ruled the land the queen was wise and kind the prince and princess played in the garden"
    w2v = Word2Vec(dim=10); w2v.train(text * 50, epochs=3)
    for word in ["king","queen","castle"]:
        print(f"Similar to '{word}': {w2v.similar(word, 3)}")
