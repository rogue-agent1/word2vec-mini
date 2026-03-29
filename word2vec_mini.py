#!/usr/bin/env python3
"""Word2Vec — minimal skip-gram with negative sampling."""
import math, random, re, sys

class Word2Vec:
    def __init__(self, dim=10, window=2, lr=0.025):
        self.dim = dim; self.window = window; self.lr = lr
        self.W = {}; self.C = {}; self.vocab = {}; self.idx2word = []
    def _init_vec(self): return [random.uniform(-0.5,0.5)/self.dim for _ in range(self.dim)]
    def _sigmoid(self, x): return 1/(1+math.exp(-max(-10,min(10,x))))
    def _dot(self, a, b): return sum(ai*bi for ai,bi in zip(a,b))
    def build_vocab(self, tokens):
        from collections import Counter
        counts = Counter(tokens)
        for w, c in counts.most_common():
            if c >= 2:
                idx = len(self.vocab); self.vocab[w] = idx; self.idx2word.append(w)
                self.W[idx] = self._init_vec(); self.C[idx] = self._init_vec()
    def train(self, tokens, epochs=5, neg_samples=5):
        filtered = [t for t in tokens if t in self.vocab]
        n = len(self.vocab)
        for epoch in range(epochs):
            loss = 0
            for i, word in enumerate(filtered):
                widx = self.vocab[word]
                start = max(0, i-self.window); end = min(len(filtered), i+self.window+1)
                for j in range(start, end):
                    if i == j: continue
                    cidx = self.vocab[filtered[j]]
                    dot = self._dot(self.W[widx], self.C[cidx])
                    sig = self._sigmoid(dot); err = 1 - sig
                    for d in range(self.dim):
                        g = err * self.lr
                        self.W[widx][d] += g * self.C[cidx][d]
                        self.C[cidx][d] += g * self.W[widx][d]
                    for _ in range(neg_samples):
                        nidx = random.randint(0, n-1)
                        if nidx == cidx: continue
                        dot = self._dot(self.W[widx], self.C[nidx])
                        sig = self._sigmoid(dot)
                        for d in range(self.dim):
                            g = -sig * self.lr
                            self.W[widx][d] += g * self.C[nidx][d]
                            self.C[nidx][d] += g * self.W[widx][d]
    def similar(self, word, top_k=5):
        if word not in self.vocab: return []
        wv = self.W[self.vocab[word]]
        scores = []
        for idx, w in enumerate(self.idx2word):
            if w == word: continue
            cos = self._dot(wv, self.W[idx]) / (math.sqrt(self._dot(wv,wv)) * math.sqrt(self._dot(self.W[idx],self.W[idx])) + 1e-8)
            scores.append((cos, w))
        scores.sort(reverse=True)
        return scores[:top_k]

if __name__ == "__main__":
    text = "the king and the queen ruled the kingdom the prince and princess lived in the castle " * 20
    text += "the cat sat on the mat the dog chased the cat the bird flew over the tree " * 20
    tokens = re.findall(r'\w+', text.lower())
    w2v = Word2Vec(dim=20, window=3); w2v.build_vocab(tokens)
    w2v.train(tokens, epochs=10)
    print(f"Vocab: {len(w2v.vocab)} words")
    for word in ["king", "cat", "the"]:
        if word in w2v.vocab:
            sims = w2v.similar(word, 5)
            print(f"  Similar to '{word}': {[(w, f'{s:.3f}') for s,w in sims]}")
