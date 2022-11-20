#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns  
import matplotlib.pyplot as plt 
import sklearn 


# In[2]:


flight_data = sns.load_dataset("flights")


# In[3]:


flight_data.head()


# In[4]:


flight_data.describe()


# In[5]:


flights = flight_data.pivot("year","month","passengers")
ax = sns.heatmap(flights)
plt.title("Heatmap of flights data")
plt.show()


# In[6]:


flights = flight_data.pivot("year","month","passengers")
plt.figure(figsize=(10, 7))
ax = sns.heatmap(flights)
plt.title("Heatmap of flights data")
plt.show()


# In[7]:


flights = flight_data.pivot("month","year","passengers")
plt.figure(figsize=(10, 7))
ax = sns.heatmap(flights)
plt.title("Heatmap of flights data")
plt.show()


# In[8]:


flights = flight_data.pivot("month","year","passengers")
plt.figure(figsize=(7, 7))
ax = sns.heatmap(flights, cbar=False)
plt.title("Heatmap of flights data")
plt.show()


# In[9]:


flights = flight_data.pivot("month","year","passengers")
plt.figure(figsize=(7, 10))
ax = sns.heatmap(flights, cbar_kws={"orientation": "horizontal"})
plt.title("Heatmap of flights data")
plt.show()


# In[10]:


flights = flight_data.pivot("month","year","passengers")
plt.figure(figsize=(10, 7))
ax = sns.heatmap(flights, linewidths=.5)
plt.title("Heatmap of flights data")
plt.show()


# In[11]:


flights = flight_data.pivot("month","year","passengers")
plt.figure(figsize=(10, 7))
ax = sns.heatmap(flights, linewidths=.5, annot=True, fmt="d")
plt.title("Heatmap of flights data")
plt.show()


# In[12]:


flights = flight_data.pivot("month","year","passengers")
plt.figure(figsize=(10, 7))
ax = sns.heatmap(flights, linewidths=.5,linecolor='black', annot=True, fmt="d")
plt.title("Heatmap of flights data")
plt.show()


# In[13]:


flights = flight_data.pivot("month","year","passengers")
plt.figure(figsize=(10, 7))
ax = sns.heatmap(flights,linewidths=.5,linecolor='black', cmap='plasma')
plt.title("Heatmap of flights data")
plt.show()


# In[15]:


from sklearn.datasets import fetch_california_housing
housing_data = fetch_california_housing()


# In[16]:


df = pd.DataFrame(housing_data.data)
df.columns = housing_data.feature_names
df['Price'] = housing_data.target
df.head()


# In[17]:


print(housing_data.DESCR)


# In[18]:


df.corr()


# In[19]:


plt.figure(figsize=(10, 7))
ax = sns.heatmap(df.corr(), annot=True)
plt.title("California housing data")
plt.show()


# In[20]:


plt.figure(figsize=(10, 7))
mask = np.triu(np.ones_like(df.corr()))
ax = sns.heatmap(df.corr(), annot=True, mask = mask)
plt.title("California housing data")
plt.show()


# In[21]:


plt.figure(figsize=(10, 7))
mask = np.triu(np.ones_like(df.corr()))
ax = sns.heatmap(df.corr(), annot=True, mask = mask, 
                 annot_kws={'fontsize':12})
plt.title("California housing data")
plt.show()


# In[22]:


plt.figure(figsize=(10, 7))
mask = np.triu(np.ones_like(df.corr()))
ax = sns.heatmap(df.corr(), annot=True, mask = mask, 
                 annot_kws={'fontsize':12}, fmt=".3f")
plt.title("California housing data")
plt.show()


# In[23]:


plt.figure(figsize=(10, 7))
mask = np.triu(np.ones_like(df.corr()))
ax = sns.heatmap(df.corr(), annot=True, mask = mask, 
                 annot_kws={'fontsize':12, 'fontweight':'bold'}, fmt=".3f")
plt.title("California housing data")
plt.show()


# In[24]:


plt.figure(figsize=(10, 7))
mask = np.triu(np.ones_like(df.corr()))
ax = sns.heatmap(df.corr(), annot=True, mask = mask, 
                 annot_kws={'fontsize':12, 'fontweight':'bold'},
                 linewidths=.5, fmt=".3f")
plt.title("California housing data")
plt.show()

