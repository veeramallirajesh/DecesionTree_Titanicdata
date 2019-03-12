#!/usr/bin/env python
# coding: utf-8

# In[180]:


import pandas as pd
df = pd.read_csv('/Users/kavya/Documents/Python_datascience/Titanic/train.csv')
df.head()


# In[181]:


df.describe()


# In[182]:


df1 = df.drop(['Name','Ticket','Cabin','Embarked','PassengerId','SibSp','Parch'],axis='columns')


# In[183]:


df1.describe()


# In[184]:


df1.dropna(inplace=True)
df1


# In[185]:


df1.describe()


# In[186]:


inputs = df1.drop('Survived',axis='columns')
target = df1['Survived']
inputs.head()


# In[187]:


target.head()


# In[188]:


inputs.describe()


# In[189]:


inputs.dtypes


# In[190]:


inputs.isnull().values.any()


# In[191]:


from sklearn.preprocessing import LabelEncoder


# In[192]:


lb = LabelEncoder()


# In[193]:


inputs['Sex_n'] = lb.fit_transform(inputs['Sex'])
inputs['Age_n'] = lb.fit_transform(inputs['Age'].astype('int64'))
inputs['Fare_n'] = lb.fit_transform(inputs['Fare'].astype('int64'))


# In[196]:


inputs.dtypes


# In[195]:


inputs.head()


# In[197]:


inputs_n=inputs.drop(['Sex','Age','Fare'],axis='columns')
inputs_n.head()


# In[198]:


from sklearn import tree


# In[199]:


model = tree.DecisionTreeClassifier()


# In[200]:


model.fit(inputs_n,target)


# In[201]:


model.score(inputs_n,target)


# In[205]:


model.predict([[1,0,22,4]])

