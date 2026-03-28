#!/usr/bin/env python3
"""Minimal Word2Vec (skip-gram with negative sampling) from scratch."""
import sys,random,math,re
from collections import Counter

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def sigmoid(x):
    return 1/(1+math.exp(-max(-500,min(500,x))))

def word2vec(corpus, dim=20, window=2, neg=5, lr=0.025, epochs=5):
    words = tokenize(corpus)
    vocab = list(set(words)); w2i = {w:i for i,w in enumerate(vocab)}
    V = len(vocab); freq = Counter(words)
    # Negative sampling distribution
    power = [freq[w]**0.75 for w in vocab]
    total = sum(power); power = [p/total for p in power]
    # Init embeddings
    W = [[random.gauss(0,0.1) for _ in range(dim)] for _ in range(V)]
    C = [[random.gauss(0,0.1) for _ in range(dim)] for _ in range(V)]
    for ep in range(epochs):
        for i,w in enumerate(words):
            wi = w2i[w]
            ctx_range = range(max(0,i-window), min(len(words),i+window+1))
            for j in ctx_range:
                if i==j: continue
                ci = w2i[words[j]]
                # Positive
                dot = sum(W[wi][d]*C[ci][d] for d in range(dim))
                g = (sigmoid(dot)-1)*lr
                for d in range(dim):
                    tw = W[wi][d]; W[wi][d]-=g*C[ci][d]; C[ci][d]-=g*tw
                # Negative samples
                for _ in range(neg):
                    ni = random.choices(range(V), weights=power, k=1)[0]
                    if ni==ci: continue
                    dot = sum(W[wi][d]*C[ni][d] for d in range(dim))
                    g = sigmoid(dot)*lr
                    for d in range(dim):
                        tw = W[wi][d]; W[wi][d]-=g*C[ni][d]; C[ni][d]-=g*tw
    return vocab, W

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(x*x for x in b))
    return dot/(na*nb) if na*nb>0 else 0

def main():
    if "--demo" in sys.argv:
        random.seed(42)
        corpus = ("the king sat on the throne. the queen sat beside the king. "
                  "a man and a woman walked in the garden. the prince is the son of the king. "
                  "the princess is the daughter of the queen. the castle has a tall tower.")*5
        vocab, W = word2vec(corpus, dim=15, epochs=10)
        w2i = {w:i for i,w in enumerate(vocab)}
        for word in ["king","queen","prince","castle"]:
            if word in w2i:
                sims = [(w, cosine(W[w2i[word]], W[w2i[w]])) for w in vocab if w!=word]
                sims.sort(key=lambda x:-x[1])
                print(f"{word}: {', '.join(f'{w}({s:.2f})' for w,s in sims[:5])}")
    else:
        corpus = sys.stdin.read()
        vocab, W = word2vec(corpus)
        print(f"Trained {len(vocab)} word vectors")
if __name__=="__main__": main()
