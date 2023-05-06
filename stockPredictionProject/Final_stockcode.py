#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


# In[2]:


company = input("Enter a string: ")
path = "C:\\Users\\91721\\OneDrive\\Desktop\\SEM4\\CSV\\"+company+".csv"
print (path)


# In[3]:


df = pd.read_csv(path)
df.head()


# In[4]:


df.dtypes
df['Date']=pd.to_datetime(df.Date)
df.shape


# In[5]:


df.drop('Adj Close',axis=1,inplace=True)
df['Volume']=df['Volume'].astype(float)
df.dtypes

df.isnull().sum()
df.isna().any()

df_new = df[np.isfinite(df).all(1)]

df_new['Open'].plot(figsize=(16,6))

x=df_new[['Open','High','Low','Volume']]
y=df_new['Close']   


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[36]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import  confusion_matrix,accuracy_score
regressor=LinearRegression()


# In[37]:


regressor.fit(x_train,y_train)

print(regressor.coef_)

print(regressor.intercept_)


# In[38]:


predicted=regressor.predict(x_test)
print(predicted)


# In[39]:


#dframe=pd.DataFrame(y_test,predicted)
#dframe.shape
dfr=pd.DataFrame({'Actual':y_test,'Predicted':predicted})


# In[40]:


print(dfr)


# In[41]:


dfr.sort_values("Actual",ascending=False)
dfr.sort_values("Predicted",ascending=False)


# In[42]:


lst=[i for i in range(0,len(dfr.Actual))]
ak=dfr.head(len(dfr.Actual))
aka=ak.sort_values("Actual")

#print(ak)
plt.plot(lst,aka)
plt.xlabel("Time")
plt.ylabel("Stock Closing Price")
plt.title("Actual Closing Price")
#plt.plot(lst,aak,color="Blue")
#df_new['Close'].plot(figsize=(16,6))
plt.show()


# In[43]:


lst=[i for i in range(0,len(dfr.Actual))]
aak=ak.sort_values("Predicted")
plt.plot(lst,aak,color="Yellow")
plt.xlabel("Time")
plt.ylabel("Stock Closing Price")
plt.title("Predicted Closing Price")
plt.show()


# In[44]:


regressor.score(x_test,y_test)


# In[ ]:




