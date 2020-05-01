# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:56:41 2019

@author: newma
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()
iris.data = np.insert(iris.data, 0, 1, axis=1)
train_y = iris.target
train_y[train_y==0]=0
train_y[train_y==1]=0
train_y[train_y==2]=1
print(train_y)

"""# New Section"""

N=150
d=iris.data.shape[1]
eta=0.1
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

rng = np.random.RandomState(12345)
w = (rng.randn(1, d)/10).ravel()
print(iris.data[0,:].shape)
print(w.shape)

for epoch in range(30):
  numcorr=0
  dw=np.zeros(w.shape)
  for t in range(N):
    r=iris.target[t]
    x=iris.data[t,:]
    y=sigmoid(np.dot(x,w))
    if (y>0.5 and r==1) or (y<=0.5 and r==0) :
      numcorr=numcorr+1
    delta=r-y
    dw=dw+x*delta
  w=w+eta*dw/N
  print(str(100*numcorr/N) + '%')