# type-2-fuzzy-c-regression-clustering-algorithm implemented in python on iris dataset
# this code is written by me from a paper called "A type-2 fuzzy c-regression clustering algorithm for Takagi–Sugeno
# system identification and its application in the steel industry" published in Information Sciences (2012)
# the authers were: M.H. Fazel Zarandi , R. Gamasaee, I.B. Turksen
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 23:49:11 2020

@author: sadegh malekshahi
"""
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
######IRIS DATASET#######
iris = datasets.load_iris()
Xdata = iris.data[:, :4]
Ydata = iris.target
Y2=np.zeros((len(Xdata),1))
for i in range(len(Xdata)):
    Y2[i][0]=Ydata[i]
Ydata=Y2
c=3#number of clusters
fuzzinesslevel=2
error = 0.005
max_iteration=100
membership=np.zeros((c,len(Xdata)))
membershipbefore=np.zeros((c,len(Xdata)))
for i in range(len(Xdata)):
    np.random.seed(1234)
    j=np.random.randint(0,c)
    membership[j][i]=1
bias=np.ones((len(Xdata),1))
#X_tilda is the same X with a 1 column
X_tilda = np.concatenate((Xdata, bias), axis=1)
objectivearray=[]
iterationarray=[]
for iteration in range (max_iteration) :
    W=[]
    regressionparameters=[] 
    for j in range(0,c): 
        w=np.zeros((len(X_tilda),len(X_tilda)))
        for i in range (len(X_tilda)):
            w[i][i]=membership[j][i]
        W.append(w)   
        regressionparameter = np.dot(np.linalg.pinv(np.dot(np.dot(X_tilda.T,W[j]), X_tilda)), np.dot(np.dot(X_tilda.T,W[j]), Ydata))
        regressionparameters.append(regressionparameter)
    f=np.zeros((c,len(Xdata)))  
    for j in range(c):
        for i in range(len(Xdata)):
            f[j][i]=np.dot(X_tilda[i].reshape(1,len(X_tilda[i])),regressionparameters[j].reshape(len(regressionparameters[j]),1))    
    errordistance=np.zeros((c,len(X_tilda)))
    for j in range (c):
        for i in range (len(Xdata)):
            errordistance[j][i]=np.abs(f[j][i]-Ydata[i])      
    ih=[]
    for i in range (0,len(Xdata)):
        for j in range(0,c) :
            if  errordistance[j][i]==0:  
               ih.append(j)         
        if len(ih)==0 :     
            for j in range(0,c) :
                s = 0
                for k in range(0,c):
                    s += ( (errordistance[j][i])/(errordistance[k][i])) ** (2 / (fuzzinesslevel - 1))
                b=1/s    
                membershipbefore[j][i] = membership[j][i]  
                membership[j][i] = 1/s             
        else :
            for j in range(c):
                 if j in ih :
                       membershipbefore[j][i] =membership[j][i]  
                       membership[j][i]=1/(len(ih)) 
                 else:
                      membershipbefore[j][i] =membership[j][i]  
                      membership[j][i]=0             
        ih=[]
    objective=0    
    for i in range(len(Xdata)):
        for j in range(c):
            objective+=(membership[j][i]**fuzzinesslevel)*(errordistance[j][i]**2)
    objectivearray.append(objective)
    iterationarray.append(iteration)
    if max(np.abs( np.array(membership).ravel()-np.array(membershipbefore).ravel() ))  <= error:
        print("number of iterations",iteration)
        break 
z=np.concatenate((Xdata,Ydata),axis=1)
z=z.T
v=[]
F=[]
FHVall=[]
for j in range(c):
    vsummation0=0
    vsummation1=0
    for i in range(len(Xdata)):
       vsummation0+=((membership[j][i])**fuzzinesslevel)*z[:,i]
       vsummation1+=((membership[j][i])**fuzzinesslevel)
    v.append((np.array(vsummation0))*(1/vsummation1))
    p=0
    FHV=0
    for i in range(len(Xdata)):
        p+=((membership[j][i])**fuzzinesslevel)*np.dot((z[:,i]-v[:][j]).reshape(len(z),1),((z[:,i]-v[:][j]).T).reshape(1,len(z)))
    F.append(p/vsummation1)
    FHV=np.sqrt(np.linalg.det(p/vsummation1))
    FHVall.append(FHV)
compactness=0    
for i in range(len(Xdata)):
    for j in range(c):
        compactness += (1/(len(Xdata)))*(FHVall[j])*((membership[j][i])**fuzzinesslevel)*((np.dot(X_tilda[i],regressionparameters[j])-Ydata[i])**2)
regressionparameters0=[]
for j in range(c):
    regressionparameters0.append(regressionparameters[j][len(Xdata.T)])
regressionparameters0=pd.DataFrame(regressionparameters0)
for j in range(c):    
    regressionparameters[j][len(Xdata.T)]=-1    
U=[]
for j in range(c):
    U.append(regressionparameters[j]/np.linalg.norm(regressionparameters[j]))
projection_lenght=np.zeros((c,c))    
for i in range(c):
    for j in range(c):
       projection_lenght[i][j]=np.abs(sum(U[i]*U[j]))
regressionparameters0=np.array(regressionparameters0)     
regressionparametersdifference=np.zeros((c,c))
for i in range(c):
    for j in range(c):       
       regressionparametersdifference[i][j]=np.abs(regressionparameters0[i]-regressionparameters0[j])       
regressionparametersdifference_max=np.max(regressionparametersdifference)       
shift_term=np.zeros((c,c))
for i in range(c):
    for j in range(c):
        shift_term[i][j]=regressionparametersdifference[i][j]/regressionparametersdifference_max
k1=0.00000005#a small value inorder to prevent nans
k2=0.00000006#a small value inorder to prevent nans
fsep=np.zeros((c,c))        
for i in range(c):
    for j in range(c):       
       fsep[i][j]=(shift_term[i][j]+k2)/(projection_lenght[i][j]+k1)    
fsepfinal=np.min(fsep) 
f_new=compactness/fsepfinal
print("clustering is done by the new articles validity index of =",f_new)  
print("clustering is done by the objective of =",objective)  
plt.plot(iterationarray, objectivearray, marker='*',color='red')
plt.xlabel('Number of iteration')
plt.ylabel('objective')
plt.show()
