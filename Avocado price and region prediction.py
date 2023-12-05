#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore') 


# In[2]:


data=pd.read_csv(r'C:\Users\asus 1\Desktop\Fliprobo\avocado.csv')
data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


data.drop('Unnamed: 0', axis=1, inplace=True)
data.head()


# In[7]:


data.type.unique()


# In[8]:


data['type'].value_counts()


# In[9]:


data['type']=data['type'].map({'conventional':0, 'organic':1})
data.info()


# In[10]:


data['region'].value_counts()


# In[11]:


data['year'].value_counts()


# In[12]:


sns.boxplot('year', 'AveragePrice', data=data)
plt.show()


# In[13]:


data['Date']=data['Date'].apply(pd.to_datetime)
data['Date']


# In[14]:


data['Month']=data['Date'].apply(lambda x:x.month)
data['Day']=data['Date'].apply(lambda x:x.day)
data.head()


# In[15]:


data.drop('Date', axis=1, inplace=True)
data.head()


# In[16]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# In[17]:


le = LabelEncoder()
data['region'] = le.fit_transform(data['region'])
data.head()


# In[18]:


data.region.unique()


# In[19]:


sns.heatmap(data.corr(), cmap='coolwarm', annot=True)


# In[20]:


sns.distplot(data["AveragePrice"],axlabel="Distribution of average price")


# In[21]:


sns.boxplot(x="type", y="AveragePrice", data=data)


# In[22]:


sns.boxplot(x="year", y="AveragePrice", data=data)


# # Model Building - Price Prediction

# In[24]:


X=data.drop('AveragePrice', axis=1)
y=data['AveragePrice']


# In[25]:


X


# In[26]:


y


# In[27]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)


# In[28]:


linreg=LinearRegression()
linreg.fit(X_train, y_train)


# In[29]:


lr_pred = linreg.predict(X_test)


# In[30]:


lr_score=r2_score(y_test, lr_pred)
lr_score


# In[31]:


lr_mse=mean_squared_error(y_test, lr_pred )
lr_mse


# In[32]:


lr_rmse=lr_mse**0.5
lr_rmse


# In[33]:


dt=DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_pred=dt.predict(X_test)
dt_pred


# In[34]:


dt_score=r2_score(y_test, dt_pred)
dt_score


# In[35]:


dt_mse=mean_squared_error(y_test, dt_pred )
dt_mse


# In[36]:


dt_rmse=dt_mse**0.5
dt_rmse


# In[37]:


rfr=RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_pred=rfr.predict(X_test)
rfr_pred


# In[38]:


rfr_score=r2_score(y_test, rfr_pred)
rfr_score


# In[39]:


rfr_mse=mean_squared_error(y_test, rfr_pred )
rfr_mse


# In[40]:


rfr_rmse=rfr_mse**0.5
rfr_rmse


# In[41]:


xgb=XGBRegressor()
xgb.fit(X_train, y_train)
xgb_pred=xgb.predict(X_test)
xgb_pred


# In[42]:


xgb_score=r2_score(y_test, xgb_pred)
xgb_score


# In[43]:


xgb_mse=mean_squared_error(y_test, xgb_pred )
xgb_mse


# In[44]:


xgb_rmse=xgb_mse**0.5
xgb_rmse


# In[45]:


price_model_df=pd.DataFrame({'Models':['Linear Regression', 'Random Forest Regressor', 'Decision Tree Regressor', ' XGBRegressor'], "Score" : [lr_score, dt_score, rfr_score, xgb_score], 'Mean Squared Error':[lr_mse, dt_mse, rfr_mse, rfr_mse], 'Root Mean Squared Error':[lr_rmse, dt_rmse, rfr_rmse, xgb_rmse]})
price_model_df


# In[57]:


#Root mean squared error is minimum and score is maximum for XGBRegressor model. To predict the average price we can use XGB Model.   


# # Model Building -Region Prediction

# In[47]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier


# In[48]:


X1=data.drop('region', axis=1)
y1=data['region']


# In[49]:


X1_train, X1_test, y1_train, y1_test=train_test_split(X1, y1, test_size=0.2, random_state=17)


# In[50]:


from sklearn.metrics import accuracy_score


# In[51]:


logreg=LogisticRegression()
logreg.fit(X1_train, y1_train)
logreg_pred = logreg.predict(X1_test)
logreg_score=accuracy_score(y1_test, logreg_pred)
logreg_score


# In[52]:


dtc=DecisionTreeClassifier()
dtc.fit(X1_train, y1_train)
dtc_pred = dtc.predict(X1_test)
dtc_score=accuracy_score(y1_test, dtc_pred)
dtc_score


# In[53]:


rfc1=RandomForestClassifier()
rfc1.fit(X1_train, y1_train)
rfc1_pred=rfc1.predict(X1_test)
rfc1_score=accuracy_score(y1_test, rfc1_pred)
rfc1_score


# In[54]:


gb=GradientBoostingClassifier()
gb.fit(X_train, y1_train)
gb_pred=gb.predict(X1_test)
gb_score=accuracy_score(y1_test, gb_pred)
gb_score


# In[55]:


adb=AdaBoostClassifier()
adb.fit(X1_train, y1_train)
adb_pred=adb.predict(X1_test)
adb_score=accuracy_score(y1_test, adb_pred)
adb_score


# In[56]:


region_model_df=pd.DataFrame({'Models':['Logistic Regression', 'Decision Tree', 'Random Forest Classifier', 'Gradient Boosting Classifier', 'AdaBoost Classifier'], 'Accuracy Score' : [logreg_score, dtc_score, rfc1_score, gb_score, adb_score]})
round(region_model_df.sort_values(by='Accuracy Score', ascending=False), 3)


# In[58]:


#Accuracy score is maximum for Random Forest model. To predict the region we can use Random Forest Model.


# In[ ]:




