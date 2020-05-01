# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:44:35 2020

@author: newma
"""

import time
#import random
import numpy as np
import operator

start_time = time.time()
dset = np.loadtxt(r"C:\Users\newma\OneDrive\Desktop\nb_score.csv",delimiter=",",skiprows=1)

#To view the dataset dimension
dset.shape

#To subset the dataset to only 100 records
dset=dset[0:100,:]

#Partition the score int 90- and 90+
for i in range(len(dset)):
    for j in range(986):
        if dset[i][985] <90:
            dset[i][985]= -90
        else:
            dset[i][985]=90

N = len(dset)
print(N)
count=0
count2 = 0
for i in range(N):
    if dset[i][985]<90:
        count += 1
    else:
        count2 += 1
print('Score less than 90 =',count)
print('Score greater than 90 =',count2)

#Correlation Distance
def correlation(arr1,arr2):
    m11=0
    m10=0
    m01=0
    m00=0
    for i in range(len(arr1)):
        if arr1[i]==1 and arr2[i]==1:
            m11 += 1
        if arr1[i]==1 and arr2[i]==0:
            m10 += 1
        if arr1[i]==0 and arr2[i]==1:
            m01 += 1
        if arr1[i]==0 and arr2[i]==0:
            m00 += 1
    
    a = (m11*m00 - m10*m01)
    b = ((m10 + m11)*(m01+m00)*(m11+m01)*(m00+m10))**0.5
    if b == 0 or a==0:
        cor_dist = 0
    else:
        cor_dist=a/b        
    return cor_dist

#Get the neighbors
def get_neighbor(train,test,k):
    dist_bucket=[]
    dist_container=[]
    
    for rows in train:
        dist = correlation(test,rows)
        dist_bucket.append((rows,dist))
    #dist_container.sort(key=lambda tup:tup[1])
    #dist_container=sort(key=operator.itemgetter(1),reverse=True)
    #reverse=Trues makes the data to be sorted from highest to lowest    
    dist_container=sorted(dist_bucket,key=operator.itemgetter(1),reverse=True)
    
    nebo=[]
    for i in range(1,k+1):
        nebo.append(dist_container[i][0]) #saves the first k number of rows 
    return nebo

#Make Prediction
def my_prediction(nebos):
    clv = []
    for x in range(len(nebos)):
        res= nebos[x][-1] #extracts the last column of each row (label)
        clv.append(res)   #saves the label in a list
        #print(clv)
    return max(set(clv), key = clv.count) #returns the element that occured more frequently in clv list

#Calculate accuracy percentage
"""def accuracy_metric(actual, predicted):
    match = 0    
    for i in range(len(actual)):
        if actual[i][-1] == predicted[i]:
            match += 1
    return (match / float(len(actual))) * 100.0"""

#Implement Confusion Matrix
def confusion_matrix(actual,predicted):
    tp=0
    fp=0
    tn=0
    fn=0
    fs=0
    
    for i in range(len(actual)):
        if actual[i][-1] == 90.0 and predicted[i]== 90.0:
            tp += 1
        elif actual[i][-1] == -90.0 and predicted[i]== 90.0:
            fp += 1
        elif actual[i][-1] == -90.0 and predicted[i]== -90.0:
            tn += 1
        elif actual[i][-1] == 90.0 and predicted[i]== -90.0:
            fn += 1
        else:
            fs += 1
            
    a=(tp + tn)
    b=(tp+tn+fp+fn)
    if a==0 or b==0:
        accu=0
    else:
        accu=a/b
    
    c=tp
    d= (tp+fp)       
    if c==0 or d==0:
        precisi=0
    else:
        precisi=c/d
    
    e=(tp+fn)
    if c==0 or e==0:
        recal=0
    else:
        recal=c/e
        
    f=2*(precisi*recal)
    g=(precisi+recal)
    
    if f==0 or g==0:
        f_scor=0
    else:
        f_scor=f/g
    
    print('TP:',tp)
    print('FP:',fp)
    print('TN:',tn)
    print('FN:',fn)
    print('Missing:',fs)
    print('Accuracy: ',accu*100)
    print('Precision: ',precisi*100)
    print('Recall: ',recal*100)
    print('F Score: ',f_scor*100)
    

def main(i):
    test_set=dset
    pred=[]
    print('K = ',i)
    for x in range(len(test_set)):
        neb=get_neighbor(dset,test_set[x],i)
        res=my_prediction(neb)
        pred.append(res)
        print('Expected {}, Predicted {}'.format(test_set[x][-1],res))
    #accuracy=accuracy_metric(test_set, pred)
    confusion_matrix(test_set,pred)
    #print('Accuracy 2:',accuracy,'percent for k=',i)
    print('\n')
        
        

for i in range(1,16,2):
    main(i)