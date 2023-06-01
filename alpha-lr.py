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
np.set_printoptions(threshold=sys.maxsize)

# read in data
trainpath = "train.csv"
testpath = "test.csv"
trainX = pd.read_csv(trainpath, usecols=[*range(1, 14), 23])
trainX = trainX.to_numpy()
trainy = pd.read_csv(trainpath, usecols=[0])
trainy = trainy.to_numpy()
testX = pd.read_csv(testpath, usecols=[*range(0, 13), 22])
testX = testX.to_numpy()

N = len(trainX)
testN = len(testX)

# change unreasonable value in test data into NaN, I give up bacause it doesn't work well
'''
for i in range(4):
    cnt = 0
    for j in range(testN):
        if testX[j][i+9] < 0:
            # testX[j][i+9]=np.nan
            cnt += 1
    print(cnt)
'''

# multivariate impute missing value
imp = IterativeImputer(max_iter=100000, random_state=0)
imp.fit(trainX)
trainX = imp.transform(trainX)
imp = IterativeImputer(max_iter=100000, random_state=0)
imp.fit(testX)
testX = imp.transform(testX)
print("impute missing values done")

# polynomial transform
poly = PolynomialFeatures(3)
trainX = poly.fit_transform(trainX)
DIM = len(trainX[0])
testX = poly.fit_transform(testX)
print("poly transform done")

BAGTIME = 10  # how many round of bagging
bagpred = [0]*testN  # sum up each test data's prediction in each bagging round
for t in range(BAGTIME):
    bagtrainX = []
    bagtrainy = []
    for i in range(N):
        ind = randrange(N)
        bagtrainX.append(trainX[ind])
        bagtrainy.append(trainy[ind])

    # linar regression with regularizer and cross validation
    # regr = ElasticNet(alpha=0.001,random_state=0)
    regr = ElasticNetCV(cv=5, n_alphas=5, alphas=[0.00001,0.0001,0.001,0.01,0.1],
                        random_state=0, max_iter=1000, tol=0.001, l1_ratio=1, n_jobs=-1, selection='random')
    # regr = ElasticNetCV(cv=5,random_state=0)
    # bagtrainy=bagtrainy.flatten()
    regr.fit(bagtrainX, bagtrainy)
    # print(regr.alpha_)

    pred = regr.predict(trainX)
    mse = 0
    mae = 0
    for i in range(N):
        mse += (pred[i]-trainy[i])*(pred[i]-trainy[i])
        mae += abs(pred[i]-trainy[i])
    print(t)
    print("train data's mse:")
    print(mse/N)
    print("train data's mae:")
    print(mae/N)

    # predict test data
    pred = regr.predict(testX)
    # adjust invalid prediction (<0 or >9) into 0 and 9, and sum up prediction
    for i in range(testN):
        if pred[i] < 0:
            pred[i] = 0
        elif pred[i] > 9:
            pred[i] = 9
        bagpred[i] += pred[i]

# average bagging prediction
for i in range(testN):
    bagpred[i] /= BAGTIME
    low=math.floor(bagpred[i])
    high=math.ceil(bagpred[i])
    if bagpred[i]-low<high-bagpred[i]:
        bagpred[i]=low
    else:
        bagpred[i]=high

# generate ids
ids = list(range(N, N+testN))

# write title in the first row
with open("pred.csv", "w") as f:
    f.write('id,Danceability\n')

# write id and prediction
with open("pred.csv", "a") as f:
    for each_id, row in zip(ids, bagpred):
        line = "%d" % each_id + ","+"%f" % row + "\n"
        f.write(line)
