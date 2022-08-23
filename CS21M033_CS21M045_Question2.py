#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import csv
import numpy as n
import pandas as pd
import matplotlib.pyplot as plt
d=pd.read_csv("classification_train_data.csv")
d


# In[2]:


d.shape


# In[3]:


d.shape[1]


# In[4]:


X=d['x'].values
Y=d['y'].values
label=d['label'].values
for i in range(len(X)):
    if(label[i]==0):
        label[i]=-1
label


# In[5]:


w=[0,0,1]


# In[6]:


conv=True
while (conv):
    conv = False
    for  i in range(len(X)):
        if( (((w[0]+w[1]*X[i]+w[2]*Y[i])>1) and (label[i]==1)) or (((w[0]+w[1]*X[i]+w[2]*Y[i])<-1) and (label[i]== -1)) ):
            pass
        else:
            conv = True
            w[0]=w[0]+label[i]
            w[1]=w[1]*X[i]+X[i]*label[i]
            w[2]=w[2]*Y[i]+Y[i]*label[i]


# In[7]:


w


# In[8]:


cmtrain = n.zeros((2,2))
for i in range(len(X)):
    if (label[i]==-1):
        plt.scatter(X[i], Y[i], marker='o',color='blue')
       
    else:
        plt.scatter(X[i], Y[i],marker='o',color='red')
        
    if ((w[0]+w[1]*X[i]+w[2]*Y[i])>=0):
        temp=1
        if(temp==label[i]):
            cmtrain[0][0]+=1
        else:
            cmtrain[1][0]+=1
            
    else:
        temp=-1
        if(temp==label[i]):
            cmtrain[1][1]+=1
        else:
            cmtrain[0][1]+=1
        
m=w[1]/w[2]
b=w[0]/w[2]
xp = n.linspace(-4,14,100)
yp = -(m*xp)-b
plt.plot(xp,yp, '-c')
plt.show()
print("confusion matrix for train data: ")
print(cmtrain)


# In[9]:


#For Test data

td=pd.read_csv("classification_test_data.csv")
td


# In[10]:


#For test data --------------------------------------------------------------------------------------------------

tX=td['x'].values
tY=td['y'].values
tlabel=td['label'].values
plabel=n.zeros(len(tX))
cmtest = n.zeros((2,2))
for i in range(len(tX)):
    if (tlabel[i]==0):
        plt.scatter(tX[i], tY[i], marker='o',color='blue')
       
    else:
        plt.scatter(tX[i], tY[i],marker='o',color='red')
        
    if ((w[0]+w[1]*tX[i]+w[2]*tY[i])>=0):
        plabel[i]=1
        if(plabel[i]==tlabel[i]):
            cmtest[0][0]+=1
        else:
            cmtest[1][0]+=1
            
    else:
        plabel[i]=0
        if(plabel[i]==tlabel[i]):
            cmtest[1][1]+=1
        else:
            cmtest[0][1]+=1

plt.plot(xp, yp, '-c')
plt.show()
print("confusion matrix for test data: ")
print(cmtest)


# In[11]:


count=0
with open('perceptron_prediction_output.csv','w',newline='') as csvfile:
    col=['x','y','True Label','Predicted Label']
    
    write=csv.DictWriter(csvfile,fieldnames=col)
   
    write.writeheader()
    
    for i in tX:
        count+=1
        write.writerow({'x':i,'y':tY[count-1],'True Label':tlabel[count-1],'Predicted Label':plabel[count-1]})


# In[ ]:





# In[ ]:




