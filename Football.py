#!/usr/bin/env python
# coding: utf-8

# In[41]:


import sqlite3
import pandas as pd
import matplotlib as plot
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
import pickle


# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[3]:


cnx = sqlite3.connect('database.sqlite')


# In[4]:


df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)


# In[5]:


df


# In[6]:


df.columns


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


print(df['overall_rating'][df['attacking_work_rate']=='stoc'].value_counts())


# In[10]:


df['attacking_work_rate'].value_counts()


# In[11]:


df.corr()


# In[12]:


print(df['preferred_foot'][df['overall_rating']==94].value_counts().plot(kind='bar'))


# In[13]:


df['overall_rating'].describe()


# In[14]:


df['attacking_work_rate'].value_counts().plot(kind='bar')


# In[15]:


df['attacking_work_rate']=df['attacking_work_rate'].fillna('None')


# In[16]:


df['gk_reflexes'].describe()


# In[17]:


df.isnull().sum()


# In[18]:


df['penalties']=df['penalties'].dropna()


# In[19]:


df=df.dropna()


# In[20]:


(df.isnull().sum())


# In[21]:


features = [
       'potential', 'crossing', 'finishing', 'heading_accuracy',
       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
       'gk_reflexes']


# In[22]:


X = df[features]
X


# In[23]:


target = ['overall_rating']


# In[24]:


from sklearn.preprocessing import StandardScaler


# In[25]:


sc=StandardScaler()


# In[26]:


sc.fit(X)


# In[27]:


scale_data=sc.transform(X)


# In[28]:


scale_data


# In[29]:


pca=PCA(n_components=4)


# In[30]:


pca.fit(scale_data)


# In[31]:


x_pca=pca.transform(scale_data)


# In[32]:


x_pca


# In[33]:


x_pca


# In[34]:


df1=pd.DataFrame(x_pca,columns=['Crossing','Finishing','Free_Kick_Accuracy','Heading_Accuracy'])


# In[35]:


df1


# In[36]:


X=df1


# In[37]:


y=df[target]


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,shuffle=True)


# In[39]:


model = LinearRegression()

model.fit(X_train, y_train)
print(
    model.score(X_train, y_train),
    model.score(X_test, y_test)
)


# In[154]:


model.coef_


# In[155]:


model.intercept_


# In[156]:


X.columns


# In[157]:


df['prediction']= model.predict(X)


# In[158]:


df['prediction'].head()


# In[159]:


fig=px.line(df,x='overall_rating',y='prediction')


# In[160]:


from math import sqrt


# In[161]:



# In[ ]:




# In[162]:


df['prediction'].describe()


# In[163]:


df['overall_rating'].describe()


# In[43]:


pickle.dump(model, open('model.pkl','wb'))


# In[ ]:




