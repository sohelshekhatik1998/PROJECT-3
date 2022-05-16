#!/usr/bin/env python
# coding: utf-8

# In[1]:


#kaggle link: https://www.kaggle.com/code/gokceayci/energy-efficiency
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import warnings 
warnings.filterwarnings("ignore")


# In[4]:


df = pd.read_excel("ENB2012_data.xlsx")
df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.isnull().values.any()


# In[8]:


df.eq(0).sum()


# In[9]:


df = df.rename(columns = { 'X1': 'Relative_Compactness',
                           'X2': 'Surface_Area',
                           'X3': 'Wall_Area',
                           'X4': 'Roof_Area',
                           'X5': 'Overall_Height',
                           'X6': 'Orientation',
                           'X7': 'Glazing_Area',
                           'X8': 'Glazing_Area_Distribution',
                           'Y1': 'Heating_Load',
                           'Y2': 'Cooling_Load'})


# In[10]:


df.head()


# In[11]:


df.describe().T


# In[12]:


# Histogram
df.hist(figsize = (30,15))
plt.show()


# In[13]:


df.plot(kind='density', subplots=True, layout=(5,2), figsize=(20, 15), sharex=False)
plt.show()


# In[14]:


plt.figure(figsize = (25,15))
sns.pairplot(df)
plt.show()


# In[15]:


df.corr()


# In[16]:


# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,cmap='YlOrRd')
plt.show()


# In[17]:


fig, axes = plt.subplots( figsize=(15, 20))
fig.suptitle('Numerik Değişkenlerin Boxplot ile Ayrkırı Gözlem Analizi')

plt.subplot(5,2,1)
sns.boxplot(x=df["Relative_Compactness"],palette='Set3')

plt.subplot(5,2,2)
sns.boxplot(x=df["Surface_Area"],palette='Set3')

plt.subplot(5,2,3)
sns.boxplot(x=df["Wall_Area"],palette='Set3')

plt.subplot(5,2,4)
sns.boxplot(x=df["Roof_Area"],palette='Set3')

plt.subplot(5,2,5)
sns.boxplot(x=df["Overall_Height"],palette='Set3')

plt.subplot(5,2,6)
sns.boxplot(x=df["Orientation"],palette='Set3')

plt.subplot(5,2,7)
sns.boxplot(x=df["Glazing_Area"],palette='Set3')

plt.subplot(5,2,8)
sns.boxplot(x=df["Glazing_Area_Distribution"],palette='Set3')

plt.subplot(5,2,9)
sns.boxplot(x=df["Heating_Load"],palette='Set3')

plt.subplot(5,2,10)
sns.boxplot(x=df["Cooling_Load"],palette='Set3')

plt.show()


# In[18]:


fig, axes = plt.subplots( figsize=(15, 20))
fig.suptitle('Numerik Değişkenlerin Boxplot ile Ayrkırı Gözlem Analizi')

plt.subplot(5,2,1)
sns.violinplot(x=df["Relative_Compactness"],palette='Set3')

plt.subplot(5,2,2)
sns.violinplot(x=df["Surface_Area"],palette='Set3')

plt.subplot(5,2,3)
sns.violinplot(x=df["Wall_Area"],palette='Set3')

plt.subplot(5,2,4)
sns.violinplot(x=df["Roof_Area"],palette='Set3')

plt.subplot(5,2,5)
sns.violinplot(x=df["Overall_Height"],palette='Set3')
plt.subplot(5,2,6)
sns.violinplot(x=df["Orientation"],palette='Set3')

plt.subplot(5,2,7)
sns.violinplot(x=df["Glazing_Area"],palette='Set3')

plt.subplot(5,2,8)
sns.violinplot(x=df["Glazing_Area_Distribution"],palette='Set3')

plt.subplot(5,2,9)
sns.violinplot(x=df["Heating_Load"],palette='Set3')

plt.subplot(5,2,10)
sns.violinplot(x=df["Cooling_Load"],palette='Set3')

plt.show()


# In[19]:



df = (df-np.min(df)) / (np.max(df) - np.min(df))
df .head(5)


# In[20]:


df.describe().T


# In[21]:


# veri setini değişkenlere ayıralım.
X = df.drop(["Heating_Load","Cooling_Load"],axis=1)
y = df[["Heating_Load","Cooling_Load"]]


# In[22]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state =42)


# In[23]:


def rmse(x,y):
    return math.sqrt(((x-y)**2).mean())


# In[24]:


def score(m):
    m.fit(x_train,y_train)
    
    print(m.score(x_train, y_train))


# In[25]:


regression_model = LinearRegression()
regression_model.fit(x_train, y_train)
score(regression_model)


# In[26]:


from sklearn.ensemble import RandomForestRegressor

randomForest = RandomForestRegressor(random_state=42, n_jobs=-1)
randomForest.fit(x_train, y_train)
randomForest.score(x_train, y_train)

