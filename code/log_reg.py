#!/usr/bin/env python
import numpy as np
import cvxpy as cp
import tf_idf as ti
import api_get_abstracts as getter
from multiprocessing import Pool

n_tot = 5000
n_train = 1000

arxiv = getter.get_stored_arxiv(n_tot)
snarxiv = getter.get_snarxiv(n_tot)

idx = np.random.permutation(n_tot)

arx_train = [arxiv[i] for i in idx[:n_train]]
snarx_train = [snarxiv[i] for i in idx[:n_train]]

arx_test = [arxiv[i] for i in idx[n_train:]]
snarx_test = [snarxiv[i] for i in idx[n_train:]]

print(len(arx_train),len(snarx_train),len(arx_test),len(snarx_test))

print("Data loaded")

vocab = np.load('big_vocab.npz')['vocab']
print("Vocab loaded")

idfs,_,_ = ti.idf(vocab,arx_train,snarx_train)
print("idfs computed")

arx_ti = ti.tf_idf(arx_train, vocab, idfs)
snarx_ti = ti.tf_idf(snarx_train, vocab, idfs)
print("Training tf-idfs computed")

arx_ti_test = ti.tf_idf(arx_test, vocab, idfs)
snarx_ti_test = ti.tf_idf(snarx_test, vocab, idfs)
print("Test tf-idfs computed")

ti_test = np.concatenate((arx_ti_test,snarx_ti_test))
Y_test = np.concatenate((-np.ones(len(arx_ti_test)),np.ones(len(snarx_ti_test))))


X = np.concatenate((arx_ti,snarx_ti))
Y = np.concatenate((-np.ones(len(arx_ti)),np.ones(len(snarx_ti))))



# CVXPY problem defn
W = cp.Variable(X.shape[1])
b = cp.Variable()
lam = cp.Parameter(nonneg=True)
obj = cp.Minimize(cp.sum(cp.logistic(-Y @ (X @ W.T + b))) + lam*cp.norm(W,2))
prob = cp.Problem(obj)
print("Problem defined")

lams = np.logspace(-8,2,100)

def tpr_fdr(l):
    lam.value = l
    prob.solve()
    
    hits = {-1: 0, 1: 0}
    misses = {-1: 0, 1: 0}

    Y_pred = np.sign(b.value + ti_test @ W.value.T)

    for i,y in enumerate(Y_test):
        if y == Y_pred[i]:
            hits[y] += 1
        else:
            misses[y] += 1
    
    tpr = hits[1]/(hits[1] + misses[1])
    fdr = misses[-1]/(hits[1] + misses[-1])
    arx_hits = hits[-1]
    snarx_hits = hits[1]
    arx_misses = misses[-1]
    snarx_misses = misses[1]
    return [tpr,fdr,arx_hits,snarx_hits,arx_misses,snarx_misses]

with Pool(processes=27) as pool:
    tpr_fdr = pool.map(tpr_fdr, lams)

tpfd = np.append(lams.reshape(-1,1),np.array(tpr_fdr),axis=1)


np.save('../data/lr_tpr_fdr.npy', tpfd)





# Y_train_pred = np.sign(b.value + X @ W.value.T)
# train_err = len(np.where(Y_train_pred != Y)[0])/len(Y)
# print(f"Train error is {train_err}")
# 
# 
# Y_pred = np.sign(b.value + ti_test @ W.value.T)
# test_err = len(np.where(Y_pred != Y_test)[0])/len(Y_test)
# print(f"Test error is {test_err}")



