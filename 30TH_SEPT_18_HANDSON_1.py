
# coding: utf-8

# In[59]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression


# In[60]:


os.chdir("C:/Users/anike/Downloads/datasets/30th Sept 18")


# In[61]:


train_df = pd.read_csv('train.csv')


# In[62]:


train_df.columns


# In[63]:


correlation_values = train_df.select_dtypes(include=[np.number]).corr()
#taking only numerical columns from the df


# In[64]:


correlation_values


# In[65]:


correlation_values[['SalePrice']]
#anything above 0.7 or 0.6 positive and negative .7 and .6 should be suppose to keep 
#OverallQual


# In[66]:


selected_features = correlation_values[["SalePrice"]][(correlation_values["SalePrice"]>=0.6)|(correlation_values["SalePrice"]<=-0.6)]


# In[67]:


selected_features


# In[68]:


a = list(selected_features.index)


# In[69]:


train_df[a].corr()


# In[85]:


x = train_df[['OverallQual','TotalBsmtSF','GrLivArea','GarageCars','GarageArea']]


# In[86]:


y= train_df['SalePrice']


# In[87]:


x_train,x_test, y_train,y_test = tts(x,y,test_size = 0.3,random_state = 42)


# In[88]:


reg = LinearRegression()


# In[89]:


model = reg.fit(x_train,y_train)


# In[90]:


y_pred = model.predict(x_test)


# In[91]:


reg.score(x_test,y_test)
#R^2 can never decrease if we add more features thus it is not r^2

