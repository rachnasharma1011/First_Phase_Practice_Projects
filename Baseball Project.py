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
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore') 


# In[2]:


data =pd.read_csv(r'C:\Users\asus 1\Desktop\Fliprobo\baseball.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.isnull().sum()


# In[6]:


data.info()


# In[7]:


data.describe()


# # Exploratory Data Analysis

# In[8]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(x='R',data=data,kde=True)
plt.subplot(1,3,2)
sns.regplot(x='R',y='W',data=data)
plt.subplot(1,3,3)
sns.boxplot(y='R',data=data)
plt.show()


# In[9]:


#Run and win are linearly correlated, data is right skewed And outliers are present between 850 and 900


# In[10]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(x='AB',data=data,kde=True)
plt.subplot(1,3,2)
sns.regplot(x='AB',y='W',data=data)
plt.subplot(1,3,3)
sns.boxplot(y='AB',data=data)
plt.show()


# In[11]:


#At balls is very weekly related to Wins With no outliers.


# In[12]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(x='H',data=data,kde=True)
plt.subplot(1,3,2)
sns.regplot(x='H',y='W',data=data)
plt.subplot(1,3,3)
sns.boxplot(y='H',data=data)
plt.show()


# In[13]:


#Hits are also very weekly related to winning with no outliers, data is right skewed.


# In[14]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(x='2B',data=data,kde=True)
plt.subplot(1,3,2)
sns.regplot(x='2B',y='W',data=data)
plt.subplot(1,3,3)
sns.boxplot(y='2B',data=data)
plt.show()


# In[15]:


#Doubles are linearly corelated with the winning and data is left skewed. 


# In[16]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(x='3B',data=data,kde=True)
plt.subplot(1,3,2)
sns.regplot(x='3B',y='W',data=data)
plt.subplot(1,3,3)
sns.boxplot(y='3B',data=data)
plt.show()


# In[17]:


#Triple has a negative corelation with winning and data is right skewed.


# In[18]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(x='HR',data=data,kde=True)
plt.subplot(1,3,2)
sns.regplot(x='HR',y='W',data=data)
plt.subplot(1,3,3)
sns.boxplot(y='HR',data=data)
plt.show()


# In[19]:


#Home run has linear relation with winning and data is right skewed


# In[20]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(x='BB',data=data,kde=True)
plt.subplot(1,3,2)
sns.regplot(x='BB',y='W',data=data)
plt.subplot(1,3,3)
sns.boxplot(y='BB',data=data)
plt.show()


# In[21]:


#Walk has linear corelation with win


# In[22]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(x='SO',data=data,kde=True)
plt.subplot(1,3,2)
sns.regplot(x='SO',y='W',data=data)
plt.subplot(1,3,3)
sns.boxplot(y='SO',data=data)
plt.show()


# In[23]:


#Strikeout has no corelation with win and the data is left-skewed


# In[24]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(x='SB',data=data,kde=True)
plt.subplot(1,3,2)
sns.regplot(x='SB',y='W',data=data)
plt.subplot(1,3,3)
sns.boxplot(y='SB',data=data)
plt.show()


# In[25]:


#Stolen Bases has negative corelation with win


# In[26]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(x='RA',data=data,kde=True)
plt.subplot(1,3,2)
sns.regplot(x='RA',y='W',data=data)
plt.subplot(1,3,3)
sns.boxplot(y='RA',data=data)
plt.show()


# In[27]:


#Run average has negative corelation with win and data is left skewed


# In[28]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(x='ER',data=data,kde=True)
plt.subplot(1,3,2)
sns.regplot(x='ER',y='W',data=data)
plt.subplot(1,3,3)
sns.boxplot(y='ER',data=data)
plt.show()


# In[29]:


#Earned runs is higly corelated to win. 


# In[30]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(x='ERA',data=data,kde=True)
plt.subplot(1,3,2)
sns.regplot(x='ERA',y='W',data=data)
plt.subplot(1,3,3)
sns.boxplot(y='ERA',data=data)
plt.show()


# In[31]:


#Earned runs average is higly corelated to win. 


# In[32]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(x='CG',data=data,kde=True)
plt.subplot(1,3,2)
sns.regplot(x='CG',y='W',data=data)
plt.subplot(1,3,3)
sns.boxplot(y='CG',data=data)
plt.show()


# In[33]:


#Complete Game is not corelated to win and the data is right skewed


# In[34]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(x='SHO',data=data,kde=True)
plt.subplot(1,3,2)
sns.regplot(x='SHO',y='W',data=data)
plt.subplot(1,3,3)
sns.boxplot(y='SHO',data=data)
plt.show()


# In[35]:


#Shutouts are linearly corelated with win and data is right skewed 


# In[36]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(x='SV',data=data,kde=True)
plt.subplot(1,3,2)
sns.regplot(x='SV',y='W',data=data)
plt.subplot(1,3,3)
sns.boxplot(y='SV',data=data)
plt.show()


# In[37]:


#Saves are linearly corelated with win


# In[38]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.histplot(x='E',data=data,kde=True)
plt.subplot(1,3,2)
sns.regplot(x='E',y='W',data=data)
plt.subplot(1,3,3)
sns.boxplot(y='E',data=data)
plt.show()


# In[39]:


#Errors are not corelated to win, data is right skewed and is right skewed


# In[40]:


plt.figure(figsize=(15,15))
sns.heatmap(data.corr(), annot=True)


# # MODEL BUILDING

# In[41]:


X=data.drop('W', axis=1)
y=data['W']


# In[42]:


X


# In[43]:


y


# In[44]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)


# In[45]:


linreg=LinearRegression()
linreg.fit(X_train, y_train)


# In[46]:


y_pred_linreg = linreg.predict(X_test)


# In[47]:


from sklearn.metrics import r2_score


# In[48]:


score_lr=r2_score(y_test, y_pred_linreg)
score_lr


# In[49]:


lr_mse=mean_squared_error(y_test, y_pred_linreg )
lr_mse


# In[50]:


lr_rmse=lr_mse**0.5
lr_rmse


# In[51]:


dt=DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_pred=dt.predict(X_test)
dt_pred


# In[52]:


dt_score=r2_score(y_test, dt_pred)
dt_score


# In[53]:


dt_mse=mean_squared_error(y_test, dt_pred )
dt_mse


# In[54]:


dt_rmse=dt_mse**0.5
dt_rmse


# In[55]:


rfr=RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_pred=rfr.predict(X_test)
rfr_pred


# In[56]:


rfr_score=r2_score(y_test, rfr_pred)
rfr_score


# In[57]:


rfr_mse=mean_squared_error(y_test, rfr_pred )
rfr_mse


# In[58]:


rfr_rmse=rfr_mse**0.5
rfr_rmse


# In[59]:


from xgboost import XGBRegressor


# In[60]:


xgb=XGBRegressor()
xgb.fit(X_train, y_train)
xgb_pred=xgb.predict(X_test)
xgb_pred


# In[61]:


xgb_score=r2_score(y_test, xgb_pred)
xgb_score


# In[62]:


xgb_mse=mean_squared_error(y_test, xgb_pred )
xgb_mse


# In[63]:


xgb_rmse=xgb_mse**0.5
xgb_rmse


# In[64]:


model_df=pd.DataFrame({'Models':['Linear Regression', 'Random Forest Regressor', 'Decision Tree Regressor', ' XGBRegressor'], "Score" : [score_lr, dt_score, rfr_score, xgb_score], 'Mean Squared Error':[lr_mse, dt_mse, rfr_mse, rfr_mse], 'Root Mean Squared Error':[lr_rmse, dt_rmse, rfr_rmse, xgb_rmse]})
model_df


# In[65]:


#The RMSE measures the standard deviation of the errors between predicted and actual values. A lower value is better. Comparing the RMSE for different models, we see that the Linear regression model provides more accurate predictions compared to other models.

