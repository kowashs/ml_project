#!/usr/bin/env python
import numpy as np
import api_get_abstracts as getter

arxiv_tot = getter.get_stored_arxiv(2000)
snarxiv_tot = getter.get_snarxiv(2000)

arxiv_train = arxiv_tot[:500]
snarxiv_train = snarxiv_tot[:500]

arxiv_test = arxiv_tot[500:]
snarxiv_test = snarxiv_tot[500:]

vocab = np.load('big_vocab.npz')['vocab']

def idf(vocab,arxiv,snarxiv): #assumes abstracts pre-parsed, but in (N,1) array
    N_arxiv = len(arxiv)
    N_snarxiv = len(snarxiv)
    N_tot = N_arxiv + N_snarxiv
    tot = np.concatenate((arxiv,snarxiv))
    dfs = np.zeros_like(vocab,dtype=float)

    for i,word in enumerate(vocab):
        for abstract in tot:
            if word in abstract:
                dfs[i] += 1

    return np.log10(N_tot/dfs)


def tf(vocab, abstract):
    counts = np.zeros_like(vocab,dtype=float)
    tfs = np.zeros_like(vocab,dtype=float)

    for i,word in enumerate(vocab):
        if word in abstract:
            counts[i] += 1
    
    for i,count in enumerate(counts):
        if count > 0:
            tfs[i] = 1 + np.log10(count)

    return tfs


def tf_idf(docs, vocab, idfs): #submit list/array of docs always
    n = len(docs)
    p = len(idfs)

    tf_idfs = np.zeros((n,p))

    for i in range(n):
        tf_idfs[i] = tf(vocab,docs[i])*idfs

    return tf_idfs



idfs = idf(vocab,arxiv_train,snarxiv_train)

tf_idfs = tf_idf(arxiv_test, vocab, idfs)

print(vocab[:50])
print(tf_idfs[0,:50])



