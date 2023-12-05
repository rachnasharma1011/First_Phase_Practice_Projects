#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv(r'C:\Users\asus 1\Desktop\Fliprobo\HR-Employee-Attrition.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data.dtypes


# In[8]:


data.describe()


# In[9]:


data['Attrition'].value_counts()


# In[10]:


sns.countplot(data['Attrition'])


# In[11]:


#Number of employess that left or stayed by age
plt.subplots(figsize=(15,5))
sns.countplot(x='Age', hue='Attrition', data=data,palette="Set3")


# In[12]:


plt.figure(figsize=(12,5))
sns.countplot(x='Department', hue='Attrition', data=data, palette="hot")
plt.title('Attrition w.r.t Department')
plt.show()


# In[13]:


plt.figure(figsize=(12,5))
sns.countplot(x='EducationField', hue='Attrition', data=data, palette="hot")
plt.title('Attrition w.r.t EducationField')
plt.show()


# In[14]:


plt.figure(figsize=(12,5))
sns.countplot(x='Gender', hue='Attrition', data=data, palette="colorblind")
plt.title('Attrition w.r.t Gender')
plt.show()


# In[15]:


plt.figure(figsize=(12,5))
sns.countplot(x='Education', hue='Attrition', data=data, palette="colorblind")
plt.title('Attrition w.r.t Education')
plt.show()


# In[16]:


plt.figure(figsize=(12,5))
sns.countplot(x='JobRole', hue='Attrition', data=data, palette="colorblind")
plt.title('Attrition w.r.t JobRole')
plt.show()


# In[17]:


plt.figure(figsize=(12,5))
sns.countplot(x='MaritalStatus', hue='Attrition', data=data, palette="colorblind")
plt.title('Attrition w.r.t MaritalStatus')
plt.show()


# In[18]:


plt.figure(figsize=(12,5))
sns.countplot(x='OverTime', hue='Attrition', data=data, palette="colorblind")
plt.title('Attrition w.r.t OverTime')
plt.show()


# In[19]:


for col in data.columns:
    if data[col].dtype==object:
        print(str(col)+':' + str(data[col].unique()))
        print(data[col].value_counts())
        print('_________________________________')


# In[20]:


data=data.drop(columns=['Over18', 'EmployeeNumber', 'StandardHours', 'EmployeeCount'], axis=1)
data.head()


# In[21]:


data.corr()


# In[22]:


plt.figure(figsize=(15,15))
sns.heatmap(data.corr(), annot=True, fmt='.0%')


# In[23]:


#Data Transformation- non numerical into numerical
from sklearn.preprocessing import LabelEncoder

for col in data.columns:
    if data[col].dtype==np.number:
        continue
    data[col]=LabelEncoder().fit_transform(data[col])


# In[24]:


data.head()


# In[25]:


#split the data
X=data.drop('Attrition',axis=1)
y=data['Attrition']


# In[26]:


X


# In[27]:


y


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=42)


# In[30]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix


# In[31]:


logreg=LogisticRegression()
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)
logreg_score=accuracy_score(y_test, logreg_pred)
logreg_score


# In[32]:


logreg_mse=mean_squared_error(y_test, logreg_pred )
logreg_mse


# In[33]:


logreg_rmse=logreg_mse**0.5
logreg_rmse


# In[34]:


dtc=DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_pred = dtc.predict(X_test)
dtc_score=accuracy_score(y_test, dtc_pred)
dtc_score


# In[35]:


dtc_mse=mean_squared_error(y_test, dtc_pred )
dtc_mse


# In[36]:


dtc_rmse=dtc_mse**0.5
dtc_rmse


# In[37]:


rfc=RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_pred=rfc.predict(X_test)
rfc_score=accuracy_score(y_test, rfc_pred)
rfc_score


# In[38]:


rfc_mse=mean_squared_error(y_test, rfc_pred )
rfc_mse


# In[39]:


rfc_rmse=rfc_mse**0.5
rfc_rmse


# In[40]:


gb=GradientBoostingClassifier()
gb.fit(X_train, y_train)
gb_pred=gb.predict(X_test)
gb_score=accuracy_score(y_test, gb_pred)
gb_score


# In[42]:


gb_mse=mean_squared_error(y_test, gb_pred )
gb_mse


# In[43]:


gb_rmse=gb_mse**0.5
gb_rmse


# In[44]:


adb=AdaBoostClassifier()
adb.fit(X_train, y_train)
adb_pred=adb.predict(X_test)
adb_score=accuracy_score(y_test, adb_pred)
adb_score


# In[45]:


adb_mse=mean_squared_error(y_test, adb_pred )
adb_mse


# In[46]:


adb_rmse=adb_mse**0.5
adb_rmse


# In[47]:


model_df=pd.DataFrame({'Models':['Logistic Regression', 'Decision Tree', 'Random Forest Classifier', 'Gradient Boosting Classifier', 'AdaBoost Classifier'], 'Accuracy Score' : [logreg_score, dtc_score, rfc_score, gb_score, adb_score], 'Mean Square Error':[logreg_mse, dtc_mse, rfc_mse, gb_mse, adb_mse], 'Root Mean Square Error':[logreg_rmse, dtc_rmse, rfc_rmse, gb_rmse, adb_rmse]})
model_df


# In[ ]:


#from the above analysis we can see that the Gradient Boosting model is giving the best accuracy and least error. 

