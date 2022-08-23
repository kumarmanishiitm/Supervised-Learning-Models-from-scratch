#!/usr/bin/env python
# coding: utf-8

# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
import csv
import numpy as n
import pandas as p
import matplotlib.pyplot as dig
#data=p.read_csv("E:\IIT Madras Course Hangout\prml tutorial\linear_reg_train_data.csv")
data=p.read_csv("linear_reg_train_data.csv")
print(data.shape)
data


# In[21]:


X=data['x'].values
Y=data['y'].values
m_x=n.mean(X)
m_y=n.mean(Y)
length=len(X)
l=0
r=0
for i in range(0,length):
    l+=(X[i]-m_x)*(Y[i]-m_y)
    r+=(X[i]-m_x)**2
m=l/r
c=m_y-m*m_x
print("slope = ",m,", bias = ",c)
print("Equation best fit line: y = ",c," + (",m,")x")


# In[22]:


max_x=n.max(X)
min_x=n.min(X)
x=n.linspace(min_x,max_x,500)
y=m*x+c
dig.plot(x,y,color='green')
dig.scatter(X,Y,color='red')
dig.xlabel("Value of X")
dig.ylabel("Value of Y")

h=0
for i in range(len(X)):
    y_b=(X[i]*m)+c
    h = h+((Y[i]-y_b)**2)
    
print("Square Error over Train data: ",h)


# In[23]:


test=p.read_csv("linear_reg_test_data.csv")
tX=test['x'].values
tY=test['y'].values
#tm_x=n.mean(tX)
#tm_y=n.mean(tY)
h=0
for i in range(len(tX)):
    y_b=(tX[i]*m)+c
    h = h+((tY[i]-y_b)**2)
    
print("Square Error over test data: ",h)

dig.plot(x,y,color='green')
dig.scatter(tX,tY,color='blue')
dig.xlabel("Value of X")
dig.ylabel("Value of Y")


# In[24]:


type(X)


# In[25]:


#degree 2

X2 = n.ones(shape=(80,3))
X2[:,1] = X
X2[:,2] = X**2
X2.shape


# In[26]:


XT2 = n.transpose(X2)
M = n.dot(XT2,X2)
Minv = n.linalg.inv(M)
W = n.dot(n.dot(Minv,XT2),Y)
W


# In[27]:


x=n.linspace(min_x,max_x,500)
y=W[0]+(W[1]*x)+(W[2]*(x**2))
dig.scatter(X,Y,color='red')
dig.plot(x,y,"g.",linewidth=1 ,markersize=2)

h=0
for i in range(len(Y)):
    Y_p=W[0]+(W[1]*X[i])+(W[2]*(X[i]**2))
    h = h+((Y[i]-Y_p)**2)
    
print("Square Error over Train data: ",h)


# In[28]:


h=0
for i in range(len(tY)):
    Y_p=W[0]+(W[1]*tX[i])+(W[2]*(tX[i]**2))
    h = h+((tY[i]-Y_p)**2)
    
print("Square Error over test data: ",h)

dig.plot(x,W[0]+(W[1]*x)+(W[2]*(x**2)),color='green')
dig.scatter(tX,tY,color='blue')
dig.xlabel("Value of X")
dig.ylabel("Value of Y")


# In[29]:


#degree 3 --------------------------------------------------------------------

X3=n.c_[X2,X**3]
XT3 = n.transpose(X3)
M = n.dot(XT3,X3)
Minv = n.linalg.inv(M)
W = n.dot(n.dot(Minv,XT3),Y)
W


# In[30]:


y=W[0]+(W[1]*x)+(W[2]*(x**2))+(W[3]*(x**3))
dig.scatter(X,Y,color='red')
dig.plot(x,y,"g.",linewidth=1 ,markersize=2)

h=0
for i in range(len(X)):
    Y_p=W[0]+(W[1]*X[i])+(W[2]*(X[i]**2))+(W[3]*(X[i]**3))
    h = h+((Y[i]-Y_p)**2)
    
print("Square Error over Train data: ",h)


# In[31]:


h=0
for i in range(len(tY)):
    Y_p=W[0]+(W[1]*tX[i])+(W[2]*(tX[i]**2))+(W[3]*(tX[i]**3))
    h = h+((tY[i]-Y_p)**2)
    
print("Square Error over test data: ",h)

dig.plot(x,W[0]+(W[1]*x)+(W[2]*(x**2))+(W[3]*(x**3)),color='green')
dig.scatter(tX,tY,color='blue')
dig.xlabel("Value of X")
dig.ylabel("Value of Y")


# In[32]:


#degree 4 --------------------------------------------------------------------

X4=n.c_[X3,X**4]
XT4 = n.transpose(X4)
M = n.dot(XT4,X4)
Minv = n.linalg.inv(M)
W = n.dot(n.dot(Minv,XT4),Y)
print(W)
y=W[0]+(W[1]*x)+(W[2]*(x**2))+(W[3]*(x**3))+(W[4]*(x**4))
dig.scatter(X,Y,color='red')
dig.plot(x,y,"g.",linewidth=1 ,markersize=2)

h=0
for i in range(len(X)):
    Y_p=W[0]+(W[1]*X[i])+(W[2]*(X[i]**2))+(W[3]*(X[i]**3))+(W[4]*(X[i]**4))
    h = h+((Y[i]-Y_p)**2)
    
print("Square Error over Train data: ",h)


# In[33]:


h=0
for i in range(len(tY)):
    Y_p=W[0]+(W[1]*tX[i])+(W[2]*(tX[i]**2))+(W[3]*(tX[i]**3))+(W[4]*(tX[i]**4))
    h = h+((tY[i]-Y_p)**2)
    
print("Square Error over test data: ",h)

y=W[0]+(W[1]*x)+(W[2]*(x**2))+(W[3]*(x**3))+(W[4]*(x**4))
dig.plot(x,y,"g.",linewidth=1 ,markersize=2)
dig.scatter(tX,tY,color='blue')
dig.xlabel("Value of X")
dig.ylabel("Value of Y")


# In[34]:


#degree 5 --------------------------------------------------------------------

X5=n.c_[X4,X**5]
XT5 = n.transpose(X5)
M = n.dot(XT5,X5)
Minv = n.linalg.inv(M)
W = n.dot(n.dot(Minv,XT5),Y)
print(W)
y=W[0]+(W[1]*x)+(W[2]*(x**2))+(W[3]*(x**3))+(W[4]*(x**4))+(W[5]*(x**5))
dig.scatter(X,Y,color='red')
dig.plot(x,y,"g.",linewidth=1 ,markersize=2)

h=0
for i in range(len(X)):
    Y_p=W[0]+(W[1]*X[i])+(W[2]*(X[i]**2))+(W[3]*(X[i]**3))+(W[4]*(X[i]**4))+(W[5]*(X[i]**5))
    h = h+((Y[i]-Y_p)**2)
    
print("Square Error over Train data: ",h)


# In[35]:


h=0
for i in range(len(tY)):
    Y_p=W[0]+(W[1]*tX[i])+(W[2]*(tX[i]**2))+(W[3]*(tX[i]**3))+(W[4]*(tX[i]**4))+(W[5]*(tX[i]**5))
    h = h+((tY[i]-Y_p)**2)
    
print("Square Error over test data: ",h)

y=W[0]+(W[1]*x)+(W[2]*(x**2))+(W[3]*(x**3))+(W[4]*(x**4))+(W[5]*(x**5))
dig.plot(x,y,"g.",linewidth=1 ,markersize=2)
dig.scatter(tX,tY,color='blue')
dig.xlabel("Value of X")
dig.ylabel("Value of Y")


# In[36]:


#degree 6 --------------------------------------------------------------------

X6=n.c_[X5,X**6]
XT6 = n.transpose(X6)
M = n.dot(XT6,X6)
Minv = n.linalg.inv(M)
W = n.dot(n.dot(Minv,XT6),Y)
print(W)
y=W[0]+(W[1]*x)+(W[2]*(x**2))+(W[3]*(x**3))+(W[4]*(x**4))+(W[5]*(x**5))+(W[6]*(x**6))
dig.scatter(X,Y,color='red')
dig.plot(x,y,"g.",linewidth=1 ,markersize=2)

h=0
for i in range(len(X)):
    Y_p=W[0]+(W[1]*X[i])+(W[2]*(X[i]**2))+(W[3]*(X[i]**3))+(W[4]*(X[i]**4))+(W[5]*(X[i]**5))+(W[6]*(X[i]**6))
    h = h+((Y[i]-Y_p)**2)
    
print("Square Error over Train data: ",h)


# In[37]:


pred = n.zeros(len(tX))
h=0
for i in range(len(tY)):
    Y_p=W[0]+(W[1]*tX[i])+(W[2]*(tX[i]**2))+(W[3]*(tX[i]**3))+(W[4]*(tX[i]**4))+(W[5]*(tX[i]**5))+(W[6]*(tX[i]**6))
    pred[i]=Y_p
    h = h+((tY[i]-Y_p)**2)
    
print("Square Error over test data: ",h)

y=W[0]+(W[1]*x)+(W[2]*(x**2))+(W[3]*(x**3))+(W[4]*(x**4))+(W[5]*(x**5))+(W[6]*(x**6))
dig.plot(x,y,"g.",linewidth=1 ,markersize=2)
dig.scatter(tX,tY,color='blue')
dig.xlabel("Value of X")
dig.ylabel("Value of Y")


# In[38]:


count=0
with open('Regression_prediction_output.csv','w',newline='') as csvfile:
    col=['x','True y','Predicted y']
    
    write=csv.DictWriter(csvfile,fieldnames=col)
   
    write.writeheader()
    
    for i in tX:
        count+=1
        write.writerow({'x':i,'True y':tY[count-1],'Predicted y':pred[count-1]})


# In[39]:


#degree 7 --------------------------------------------------------------------

X7=n.c_[X6,X**7]
XT7 = n.transpose(X7)
M = n.dot(XT7,X7)
Minv = n.linalg.inv(M)
W = n.dot(n.dot(Minv,XT7),Y)
print(W)
y=W[0]+(W[1]*x)+(W[2]*(x**2))+(W[3]*(x**3))+(W[4]*(x**4))+(W[5]*(x**5))+(W[6]*(x**6))+(W[7]*(x**7))
dig.scatter(X,Y,color='red')
dig.plot(x,y,"g.",linewidth=1 ,markersize=2)

h=0
for i in range(len(X)):
    Y_p=W[0]+(W[1]*X[i])+(W[2]*(X[i]**2))+(W[3]*(X[i]**3))+(W[4]*(X[i]**4))+(W[5]*(X[i]**5))+(W[6]*(X[i]**6))+(W[7]*(X[i]**7))
    h = h+((Y[i]-Y_p)**2)
    
print("Square Error over Train data: ",h)


# In[40]:


h=0
for i in range(len(tY)):
    Y_p=W[0]+(W[1]*tX[i])+(W[2]*(tX[i]**2))+(W[3]*(tX[i]**3))+(W[4]*(tX[i]**4))+(W[5]*(tX[i]**5))+(W[6]*(tX[i]**6))+(W[7]*(tX[i]**7))
    h = h+((tY[i]-Y_p)**2)
    
print("Square Error over test data: ",h)

y=W[0]+(W[1]*x)+(W[2]*(x**2))+(W[3]*(x**3))+(W[4]*(x**4))+(W[5]*(x**5))+(W[6]*(x**6))+(W[7]*(x**7))
dig.plot(x,y,"g.",linewidth=1 ,markersize=2)
dig.scatter(tX,tY,color='blue')
dig.xlabel("Value of X")
dig.ylabel("Value of Y")


# In[ ]:




