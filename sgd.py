import numpy as np
from liblinear.liblinearutil import *
import sys
import math
import random
import os
import time
import pandas as pd
import sklearn
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from random import randrange
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
np.set_printoptions(threshold=sys.maxsize)

#read in data
trainpath="train.csv"
testpath="test.csv"
trainX=pd.read_csv(trainpath,usecols=[*range(1,14),23])
#trainX.info()
trainX=trainX.to_numpy()
trainy=pd.read_csv(trainpath,usecols=[0])
trainy=trainy.to_numpy()
testX=pd.read_csv(testpath,usecols=[*range(0,13),22])
#testX.info()
testX=testX.to_numpy()
N=len(trainX)
testN=len(testX)

imp = IterativeImputer(max_iter=100000, random_state=0)
imp.fit(trainX)
trainX=imp.transform(trainX)
imp = IterativeImputer(max_iter=100000, random_state=0)
imp.fit(testX)
testX=imp.transform(testX)
print("impute missing values done")

#polynomial transform
poly = PolynomialFeatures(3)
trainX=poly.fit_transform(trainX)
DIM=len(trainX[0])
testX=poly.fit_transform(testX)
print("poly transform done")

reg = make_pipeline(StandardScaler(),SGDRegressor(loss='epsilon_insensitive',max_iter=1000, tol=1e-3,epsilon=0,penalty='l1',alpha=0.1,l1_ratio=1))
trainy=trainy.flatten()
reg.fit(trainX, trainy)
pred=reg.predict(testX)

for i in range(testN):
    if pred[i]<0:
        pred[i]=0
    elif pred[i]>9:
        pred[i]=9

ids=list(range(N,N+testN))
with open("sgdpred.csv","w") as f:
    f.write('id,Danceability\n')

with open("sgdpred.csv", "a") as f:
    for each_id, row in zip(ids,pred):
        line = "%d" %each_id + ","+"%f" %row + "\n"
        f.write(line)
