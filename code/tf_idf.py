#!/usr/bin/env python
import numpy as np
import api_get_abstracts as getter

arxiv_tot = getter.get_stored_arxiv(1000)
snarxiv_tot = getter.get_snarxiv(1000)

arxiv_train = arxiv_tot[:500]
snarxiv_train = snarxiv_tot[:500]

arxiv_test = arxiv_tot[500:]
snarxiv_test = snarxiv_tot[500:]

vocab = np.load('big_vocab.npz')['vocab']

def idf(vocab,arxiv,snarxiv): #assumes abstracts pre-parsed, but in (N,1) array
    N_arxiv = len(arxiv)
    N_snarxiv = len(snarxiv)
    N_tot = N_arxiv + N_snarxiv

    dfs = np.zeros(vocab.shape)
    dfs_arx = np.zeros(vocab.shape)
    dfs_snarx = np.zeros(vocab.shape)


    for i,word in enumerate(vocab):
        for abstract in arxiv:
            if word in abstract:
                dfs[i] += 1
                dfs_arx += 1
        for abstract in snarxiv:
            if word in abstract:
                dfs[i] += 1
                dfs_snarx +=1
    
    idfs = np.log10((N_tot+1)/(dfs+1))
    idfs_arx = np.log10((N_arxiv+1)/(dfs_arx+1))
    idfs_snarx = np.log10((N_snarxiv+1)/(dfs_snarx+1))
    return idfs,idfs_arx,idfs_snarx


def tf(vocab, abstract):
    counts = np.zeros(vocab.shape)
    tfs = np.zeros(vocab.shape)

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



idfs,_,_ = idf(vocab,arxiv_train,snarxiv_train)

tf_idfs = tf_idf(arxiv_test, vocab, idfs)

print(vocab[:50])
print(tf_idfs[0,:50])



