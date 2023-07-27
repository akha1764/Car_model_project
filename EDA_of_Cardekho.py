#!/usr/bin/env python
# coding: utf-8

# # EDA of Data From CarDekho.com 

# This dataset contains information about used cars listed on www.cardekho.com
# 
# The columns in the given dataset are as follows:
# 
# name
# 
# year
# 
# selling_price
# 
# km_driven
# 
# fuel
# 
# seller_type
# 
# transmission
# 
# Owner

# In[1]:


#importing Required library 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importing plotly Library
from plotly.offline import iplot
import plotly as py
import plotly.tools as tls
import cufflinks as cf
py.offline.init_notebook_mode(connected=True) #Turning on notebook mode 
cf.go_offline()


# In[3]:


df=pd.read_csv(r"CAR DETAILS FROM CAR DEKHO.csv") #dataset


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df.isnull().sum()


# In[10]:


df['Age'] = 2022 - df['year']
df.drop('year',axis=1,inplace = True)


# In[11]:


df.head()


# In[12]:


pd.set_option("display.max_rows", None, "display.max_columns",None)
df


# In[68]:


df.describe()


# In[13]:


cat_cols = ['fuel','seller_type','transmission','owner']
i=0
while i < 4:
    fig = plt.figure(figsize=[20,5])   
    
    plt.subplot(1,2,1)
    sns.countplot(x=cat_cols[i], data=df)
    i += 1  
    
    plt.subplot(1,2,2)
    sns.countplot(x=cat_cols[i], data=df)
    i += 1
    
    plt.show()


# 1. Number of petrol car & Disel car almost equal
# 2. CNG LPG & Electric Car have very few listing 
# 3. Most of the listing are by Individual then followed by dealer 
# 4. listing of manual cars are very high as compared to automatic 
# 5. listing of first owner car is high then follwed by second owner 

# In[14]:


df.info()


# In[15]:


num_cols = ['selling_price','km_driven','Age','selling_price']
i=0
while i < 4:
    fig = plt.figure(figsize=[13,3])
    
    plt.subplot(1,2,1)
    sns.boxplot(x=num_cols[i], data=df)
    i += 1
    
    
    plt.subplot(1,2,2)
    sns.boxplot(x=num_cols[i], data=df)
    i += 1
    
    plt.show()


# Dataset has lots of outliers 

# In[16]:


df['selling_price'].quantile(0.99)


# # Outliers in selling price

# In[17]:


df[df['selling_price'] > df['selling_price'].quantile(0.99)].sort_values(by="selling_price",ascending=False)


# Dataset also has  premium cars 
# 
# Highest Selling price is  89,00,000

# In[18]:


df[df['km_driven'] > df['km_driven'].quantile(0.99)].sort_values(by="km_driven",ascending=False)


# Highest km_driven is 8,06,599	

# In[19]:


df[df['Age'] > df['Age'].quantile(0.99)].sort_values(by="Age",ascending=False)


# Highest age of car is 29 Years 

# In[20]:


df.sort_values(by="selling_price")


# In[21]:


df["name"].value_counts().sort_values(ascending=False).head(10).iplot(kind="bar")


# Maruti swift Dzire VDI have the highest listing in used cars 

# In[22]:


fuel=df.groupby(by="fuel")


# In[23]:


fuel.selling_price.mean().sort_values(ascending=False).iplot(kind="bar")


# Disel Car have highest average selling price

# In[24]:


sns.heatmap(df.corr(), annot=True, cmap="RdBu")
plt.show()


# In[25]:


owner=df.groupby(by="owner")


# In[26]:


owner.selling_price.mean().sort_values(ascending=False).iplot(kind="bar")


# In[27]:


sns.pairplot(df)


# In[28]:


df.shape


# In[29]:


car_names=df["name"].unique()


# In[30]:


df["name"].nunique()


# In[31]:


pd.DataFrame(car_names)


# In[32]:


df['brand'] = df['name'].str.split(' ').str[0]


# In[33]:


df


# In[34]:


df.brand.unique()


# In[35]:


df.brand.nunique()


# In[36]:


df.brand.value_counts().sort_values(ascending=False).iplot(kind="bar")


# In[37]:


df.brand.value_counts().sort_values().head(15).iplot(kind="bar")


# In[38]:


df[df.selling_price>2500000].sort_values(by="selling_price").brand.unique()


# # Analysis of premium segment cars

# In[39]:


top_brands=['Jaguar', 'Audi', 'BMW','Land','Mercedes-Benz', 'Volvo']


# In[40]:


df[df["brand"] == 'Audi']


# In[41]:


tdf=df.loc[df['brand'].isin(top_brands)]


# In[42]:


tdf.shape


# In[43]:


df.info()


# In[44]:


sns.relplot(x=df.selling_price,y=df.km_driven,hue=df.brand)


# In[45]:


sns.relplot(x=tdf.selling_price,y=tdf.km_driven,hue=tdf.brand)


# In[46]:


tdf.brand.value_counts().sort_values(ascending=False).iplot(kind="bar")


# Audi BMW & Mercedes-Benz listing are higher in premium cars.

# In[47]:


topcars=tdf.groupby("brand")


# In[48]:


topcars.mean().sort_values(by="selling_price").iplot(kind="bar")


# Averge selling price of audi is lowest compared other premium car

# In[49]:


topcars.mean().sort_values(by="selling_price").sort_values(by="selling_price")


# In[50]:


tdf.transmission.value_counts()


# In[51]:


tdf.fuel.value_counts().plot.pie(fontsize = 18, autopct = '%.2f')


# most of  cars are diesel in premium cars 

# In[52]:


tdf.seller_type.value_counts().plot.pie(fontsize = 18, autopct = '%.2f')


# Majority of premium cars are sold by Dealers 

# In[53]:


tdf.name.value_counts().sort_values(ascending=False).head()


# This  cars have maximum number of listing in premium car segment.

# # Analysis of Normal Segment cars 

# In[54]:


ndf=df.loc[~df['brand'].isin(top_brands)]


# In[55]:


ndf


# In[56]:


ndf.info()


# In[57]:


print("Share of Normal Segment Cars in %: ") 
print(len(ndf)/len(df)*100)
print("-------------------------------------")
print("Share of premium Segment Cars in % : ") 
print(len(tdf)/len(df)*100)


# In[58]:


sns.relplot(x=ndf.selling_price,y=ndf.km_driven,hue=ndf.brand)


# In[59]:


ndf.brand.value_counts().sort_values(ascending=False).iplot(kind="bar")


# Maruti & Hyundai are most popular brands in Normal Car Segment 

# In[60]:


normalcars=ndf.groupby("brand")


# In[61]:


normalcars.mean().sort_values(by="selling_price",ascending=False).iplot(kind="bar")


# In[62]:


normalcars.mean().sort_values(by="selling_price").sort_values(by="selling_price")


# In[63]:


ndf.transmission.value_counts()


# In[64]:


ndf.seller_type.value_counts().plot.pie(fontsize = 18, autopct = '%.2f')


# In[65]:


ndf.fuel.value_counts()


# In[70]:


ndf.fuel.value_counts().plot.pie(fontsize = 10, autopct = '%.2f')


# In[67]:


ndf.fuel.value_counts().plot.bar()


# In[ ]:




