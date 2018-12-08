#!/usr/bin/env python
import numpy as np
import cvxpy as cp
import tf_idf as ti
import api_get_abstracts as getter

arxiv = getter.get_stored_arxiv(1000)
snarxiv = getter.get_snarxiv(1000)

idx = np.random.permutation(1000)

arx_train = [arxiv[i] for i in idx[:800]]
snarx_train = [snarxiv[i] for i in idx[:800]]

arx_test = [arxiv[i] for i in idx[800:]]
snarx_test = [snarxiv[i] for i in idx[800:]]

test = arx_test.extend(snarx_test)
print("Data loaded")

vocab = np.load('big_vocab.npz')
print("Vocab loaded")

idfs = ti.idf(vocab,arx_train,snarx_train)
arx_ti = ti.tf_idf(arx_train, vocab, idfs)
snarx_ti = ti.tf_idf(snarx_train, vocab, idfs)

arx_ti_test = ti.tf_idf(arx_test, vocab, idfs)
snarx_ti_test = ti.tf_idf(snarx_test, vocab, idfs)
print("tf-idfs computed")

ti_test = np.concatenate((arx_ti_test,snarx_ti_test))
Y_test = np.concatenate((-np.ones(len(arx_ti)),np.ones(len(snarx_ti))))


X = np.concatenate((arx_ti,snarx_ti))
Y = np.concatenate((-np.ones(len(arx_ti)),np.ones(len(snarx_ti))))




# CVXPY problem defn
W = cp.Variable(X.shape[1])
b = cp.Variable()
lam = cp.Parameter(nonneg=True)
obj = cp.Minimize(cp.sum(cp.logistic(-Y*X*W.T+b)) + lam*cp.norm(W,2))
prob = cp.Problem(obj)
print("Problem defined")

lam.value = 1e-4
prob.solve()
print("Problem solved")

Y_train_pred = np.sign(b.value + X*W.value.T)
train_err = len(np.where(Y_train_pred != Y))/len(Y)
print(f"Train error is {train_err}")


Y_pred = np.sign(b.value + ti_test*W.value.T)
test_err = len(np.where(Y_pred != Y_test))/len(Y_test)
print(f"Test error is {test_err}")



