#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle



# In[2]:


train=pd.read_csv('sales_train.csv')
items=pd.read_csv('items.csv')
shops=pd.read_csv('shops.csv')
cats=pd.read_csv('item_categories.csv')


# In[3]:


sns.boxplot(train.item_cnt_day)


# In[4]:


train = train[train.item_cnt_day < 1000]


# In[5]:


sns.boxplot(train.item_price)


# In[6]:


train = train[train.item_price < 300000]


# In[7]:


train = train[train.item_price > 0].reset_index(drop = True)
train.loc[train.item_cnt_day < 1, "item_cnt_day"] = 0


# In[8]:


train.loc[train.shop_id == 0, 'shop_id'] = 57
train.loc[train.shop_id == 1, 'shop_id'] = 58
train.loc[train.shop_id == 10, 'shop_id'] = 11
train.loc[train.shop_id == 40, 'shop_id'] = 39


# In[9]:


shops.loc[ shops.shop_name == 'Сергиев Посад ТЦ "7Я"',"shop_name" ] = 'СергиевПосад ТЦ "7Я"'
shops["city"] = shops.shop_name.str.split(" ").map( lambda x: x[0] )
shops["category"] = shops.shop_name.str.split(" ").map( lambda x: x[1] )
shops.loc[shops.city == "!Якутск", "city"] = "Якутск"


# In[10]:


category = []
for cat in shops.category.unique():
    if len(shops[shops.category == cat]) >= 5:
        category.append(cat)
shops.category = shops.category.apply( lambda x: x if (x in category) else "other" )


# In[11]:


from sklearn.preprocessing import LabelEncoder
shops["shop_category"] = LabelEncoder().fit_transform( shops.category )
shops["shop_city"] = LabelEncoder().fit_transform( shops.city )
shops = shops[["shop_id", "shop_category", "shop_city"]]
shops=shops.drop([0,1,10,40]).reset_index(drop=True)


# In[12]:


cats["type_code"] = cats.item_category_name.apply( lambda x: x.split(" ")[0] ).astype(str)
cats.loc[ (cats.type_code == "Игровые")| (cats.type_code == "Аксессуары"), "category" ] = "Игры"


# In[13]:


category = []
for cat in cats.type_code.unique():
    if len(cats[cats.type_code == cat]) >= 5: 
        category.append( cat )
cats.type_code = cats.type_code.apply(lambda x: x if (x in category) else "etc")


# In[14]:


cats.type_code = LabelEncoder().fit_transform(cats.type_code)
cats["split"] = cats.item_category_name.apply(lambda x: x.split("-"))
cats["subtype"] = cats.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats["subtype_code"] = LabelEncoder().fit_transform( cats["subtype"] )
cats = cats[["item_category_id", "subtype_code", "type_code"]]


# In[15]:


import re
def name_correction(x):
    x = x.lower() # all letters lower case
    x = x.partition('[')[0] # partition by square brackets
    x = x.partition('(')[0] # partition by curly brackets
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x) # remove special characters
    x = x.replace('  ', ' ') # replace double spaces with single spaces
    x = x.strip() # remove leading and trailing white space
    return x


# In[16]:


items["name1"], items["name2"] = items.item_name.str.split("[", 1).str
items["name1"], items["name3"] = items.item_name.str.split("(", 1).str

items["name2"] = items.name2.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
items["name3"] = items.name3.str.replace('[^A-Za-z0-9А-Яа-я]+', " ").str.lower()

items = items.fillna('0')

items["item_name"] = items["item_name"].apply(lambda x: name_correction(x))

items.name2 = items.name2.apply( lambda x: x[:-1] if x !="0" else "0")


# In[17]:


items["type"] = items.name2.apply(lambda x: x[0:8] if x.split(" ")[0] == "xbox" else x.split(" ")[0] )
items.loc[(items.type == "x360") | (items.type == "xbox360") | (items.type == "xbox 360") ,"type"] = "xbox 360"
items.loc[ items.type == "", "type"] = "mac"
items.type = items.type.apply( lambda x: x.replace(" ", "") )
items.loc[ (items.type == 'pc' )| (items.type == 'pс') | (items.type == "pc"), "type" ] = "pc"
items.loc[ items.type == 'рs3' , "type"] = "ps3"


# In[18]:


group_sum = items.groupby(["type"]).agg({"item_id": "count"})
group_sum = group_sum.reset_index()
drop_cols = []
for cat in group_sum.type.unique():
    if group_sum.loc[(group_sum.type == cat), "item_id"].values[0] <40:
        drop_cols.append(cat)
items.name2 = items.name2.apply( lambda x: "other" if (x in drop_cols) else x )
items = items.drop(["type"], axis = 1)


# In[19]:


items.name2 = LabelEncoder().fit_transform(items.name2)
items.name3 = LabelEncoder().fit_transform(items.name3)

items.drop(["item_name", "name1"],axis = 1, inplace= True)


# In[20]:


price=train[['item_id','item_price']]
price=price.groupby(['item_id']).mean()
price=price.reset_index()

items=pd.merge(items,price,on=['item_id'],how='inner')
items=pd.merge(items,cats,on=['item_category_id'],how='inner')


# In[21]:


train=train.drop('date',axis=1)
train=train.groupby(['date_block_num', 'shop_id', 'item_id']).sum()
train.drop('item_price',axis=1,inplace=True)
train=train.reset_index()
train=train.rename(columns = {'item_cnt_day' : 'item_cnt_month'})


# In[22]:
    
from itertools import product
matrix = []
cols  = ["date_block_num", "shop_id", "item_id"]
for i in range(34):
    sales = train[train.date_block_num == i]
    matrix.append( np.array(list( product( [i], sales.shop_id.unique(), sales.item_id.unique() ) ), dtype = np.int16) )

matrix = pd.DataFrame( np.vstack(matrix), columns = cols )
matrix["date_block_num"] = matrix["date_block_num"].astype(np.int8)
matrix["shop_id"] = matrix["shop_id"].astype(np.int8)
matrix["item_id"] = matrix["item_id"].astype(np.int16)
matrix.sort_values( cols, inplace = True )

train=pd.merge(matrix,train,on=["date_block_num", "shop_id", "item_id"],how='left')
train=train.fillna(0)
train=pd.merge(train,items,on=['item_id'],how='inner')
train=pd.merge(train,shops,on=['shop_id'],how='inner')
train.sort_values(["date_block_num", "shop_id", "item_id"], inplace = True )
train=train.reset_index(drop=True)

# In[23]:


train["month"] = train["date_block_num"] % 12
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
train["days"] = train["month"].map(days).astype(np.int8)
train["year"] = 2013+(train["date_block_num"]/12).astype(np.int16)
train['month']=train['month']+1


# In[24]:


train=train[['date_block_num', 'shop_id', 'item_id', 'shop_category', 'shop_city',
       'item_category_id', 'name2', 'name3', 'subtype_code', 'type_code', 'days', 'month', 'year',
       'item_price', 'item_cnt_month']]

items.to_csv('items_data.csv')
shops.to_csv('shops_data.csv')


# In[25]:


X_train = train[train.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = train[train.date_block_num < 33]['item_cnt_month']
X_valid = train[train.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = train[train.date_block_num == 33]['item_cnt_month']


# In[26]:


Y_train = Y_train.clip(0, 20)
Y_valid = Y_valid.clip(0, 20)


# In[27]:


from xgboost import XGBRegressor
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# In[28]:


model = XGBRegressor(
    max_depth=10,
    n_estimators=1000,
    min_child_weight=0.5, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.1,
#     tree_method='gpu_hist',
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 20)

pickle.dump(model,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))

