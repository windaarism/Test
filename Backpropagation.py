# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 08:22:24 2017

@author: Hans
"""

import numpy as np
import pandas as pd
from scipy import stats

def normalized(x):    
    nData = len(x)
    
    #z-score
    aZ = stats.zscore(x,ddof=1)
    
    #normalize
    b = aZ-(np.ones((nData,1))*aZ.min(axis=0))
    c = aZ.max(axis=0) - aZ.min(axis=0)
    x = b/c
    return x

n = pd.read_csv('titanicANN.csv')
x = n[['Survived','Pclass','Sex','Age','Fare','SibSp','Embarked']]
x = x.fillna(0)

gender = pd.get_dummies(x['Sex'])
embarked = pd.get_dummies(x['Embarked'])

y = x['Survived']
y= y.apply(lambda i: "Survived" if i == 1 else "Dead")
y = pd.get_dummies(y)
out = y.copy()
t = np.array(y.values.tolist())

x = pd.concat([x['Pclass'],gender,x['Age'],x['Fare'],x['SibSp'],embarked],axis=1)
x = normalized(x)

#layer info (input,hidden,output)
(nData,lx) = np.shape(x) #nData = banyak data; lx=input layer
ly = len(y.loc[0]) #output layer
lh = 10 #hidden layer
N = 889

#inisialisasi nguyen-widrow
beta0 = 0.7*lh**(1/float(lx))
v = np.random.uniform(low=-0.5, high=0.5, size=(lx,lh))
w = np.random.uniform(low=-0.5, high=0.5, size=(lh,ly))
v0 = np.random.uniform(low=-beta0, high=beta0, size=(lh))
w0 = np.random.uniform(low=-beta0, high=beta0, size=(ly))

norm_v = np.zeros(lh)
for j in range(lh):
    for i in range(lx):
        norm_v[j] = norm_v[j] + v[i][j]**2
    v[i][j] = (beta0/np.sqrt(norm_v[j]))*v[i][j]
    
norm_w = np.zeros(ly)
for j in range(ly):
    for i in range(lh):
        norm_w[j] = norm_w[j] + w[i][j]**2
    w[i][j] = (beta0/np.sqrt(norm_w[j]))*w[i][j]
    
MSE = 1
epoch = 0
alpha = 0.2
momentum = 0.5

z_in = np.zeros((lh,1))
z = np.zeros((lh,1))
y_in = np.zeros((ly,1))
y = np.zeros((ly,1))
deltaZ = np.zeros((lh,1))
deltaV = np.zeros((lx,lh))
deltaV0 = np.zeros((lh,1))
deltaW = np.zeros((lh,ly))
deltaW0 = np.zeros(ly)
delta_in = np.zeros((lh,1))
square_error = np.zeros((N,ly))
Y = np.zeros((len(out)-N,ly))

while (MSE > 0.0001) and (epoch <1000):    
    for n in range(N):
        for i in range(lh):
           z_in[i] = v0[i] + np.sum(x[n] * v[:,i])
           z[i] = 1/(1 + np.exp(-z_in[i]))
        for j in range(ly):
           y_in[j] = w0[j] + np.sum(z.transpose() * w[:,j])
           y[j] = 1/(1 + np.exp(-y_in[j]))
        deltaY = (t[n]-y.transpose()) * y.transpose() * (1-y.transpose())
        for j in range(len(z)):
            deltaW[j] = alpha*(deltaY * z[j]) + momentum * deltaW[j]
        deltaW0 = alpha * deltaY + momentum * deltaW0
        for j in range(lh):
            delta_in[j] = np.sum(deltaY * w[j])
        for j in range(lh):
            deltaZ[j] = delta_in[j] * z[j] * (1-z[j])
        for j in range(lh):
            deltaV[:,j] = alpha * deltaZ[j] * x[j] + momentum * deltaV[:,j]
            deltaV0[j] = alpha * deltaZ[j] + momentum * deltaV0[j]
        w = w + deltaW
        w0 = w0 + np.reshape(deltaW0,(ly))
        v = v + deltaV
        v0 = v0 + np.reshape(deltaV0.transpose(),(lh))
        square_error[n] = ((t[n] - y.transpose())**2)
    MSE = np.sum(square_error)/(N*2)
    print epoch,':',MSE
    epoch += 1

count = 0
for n in range(N,len(out)):
    for i in range(lh):
       z_in[i] = v0[i] + np.sum(x[n] * v[:,i])
       z[i] = 1/(1 + np.exp(-z_in[i]))
    for j in range(ly):
       y_in[j] = w0[j] + np.sum(z.transpose() * w[:,j])
       y[j] = 1/(1 + np.exp(-y_in[j]))
    Y[count] = y.transpose()
    count += 1

target = t[N::]
target = pd.DataFrame(data = target, columns=['tDead','tSurvived'])

prediction = pd.DataFrame(data = Y, columns=['pDead','pSurvived'])
prediction['pDead'] = prediction['pDead'].apply(lambda x: 1 if x>0.5 else 0)
prediction['pSurvived'] = prediction['pSurvived'].apply(lambda x: 1 if x>0.5 else 0)

res = pd.concat([target,prediction], axis=1)
match = np.sum((res['tDead'] == res['pDead']) & (res['tSurvived'] == res['pSurvived']))
recrate = float(match)/float(len(target))
print recrate

res.to_csv('outputBP.csv',index=False,header=True)