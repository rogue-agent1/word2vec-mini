#!/usr/bin/env python3
"""word2vec_mini - Minimal word2vec (skip-gram) implementation."""
import sys,math,random,re
from collections import defaultdict
def build_vocab(text,min_count=1):
    words=re.findall(r"\w+",text.lower());freq=defaultdict(int)
    for w in words:freq[w]+=1
    vocab={w:i for i,(w,c) in enumerate(sorted(freq.items())) if c>=min_count}
    return vocab,words
def skipgram_pairs(words,vocab,window=2):
    pairs=[]
    for i,w in enumerate(words):
        if w not in vocab:continue
        for j in range(max(0,i-window),min(len(words),i+window+1)):
            if i!=j and words[j] in vocab:pairs.append((vocab[w],vocab[words[j]]))
    return pairs
def train(pairs,vocab_size,dim=10,lr=0.01,epochs=50):
    W=[[random.gauss(0,0.1) for _ in range(dim)] for _ in range(vocab_size)]
    C=[[random.gauss(0,0.1) for _ in range(dim)] for _ in range(vocab_size)]
    for _ in range(epochs):
        random.shuffle(pairs)
        for w,c in pairs:
            dot=sum(W[w][d]*C[c][d] for d in range(dim))
            sig=1/(1+math.exp(-max(-10,min(10,dot))))
            err=sig-1
            for d in range(dim):
                gw=err*C[c][d];gc=err*W[w][d]
                W[w][d]-=lr*gw;C[c][d]-=lr*gc
    return W
def similar(W,vocab,word,n=5):
    idx=vocab[word];dists=[]
    for w,i in vocab.items():
        if i==idx:continue
        dot=sum(W[idx][d]*W[i][d] for d in range(len(W[0])))
        na=math.sqrt(sum(x**2 for x in W[idx]));nb=math.sqrt(sum(x**2 for x in W[i]))
        sim=dot/(na*nb) if na*nb>0 else 0;dists.append((w,sim))
    return sorted(dists,key=lambda x:-x[1])[:n]
if __name__=="__main__":
    text="the king and the queen live in the castle the prince and princess play in the garden"
    vocab,words=build_vocab(text);pairs=skipgram_pairs(words,vocab)
    W=train(pairs,len(vocab),dim=8,epochs=100)
    rev={i:w for w,i in vocab.items()}
    for word in["king","queen","castle"]:
        if word in vocab:
            sims=similar(W,vocab,word);print(f"Similar to '{word}': {[(w,round(s,2)) for w,s in sims]}")
