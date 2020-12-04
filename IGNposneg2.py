#!/usr/bin/env python
# coding: utf-8

# IGN review classification by BERT 

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/minman28/QTM-340/main/IGNposneg2.csv', delimiter=',',index_col=0)


# In[3]:


df


# In[4]:


df.groupby('score').count()


# In[5]:


#I sampled 100 negative and positive reviews to create the final dataframe for this test
df_pos = df[df['score'] == 1]
df_neg = df[df['score'] == 0]
df_neg = df_neg.sample(n=100)
df_pos = df_pos.sample(n=100)


# In[6]:


df_final = pd.concat([df_pos, df_neg])
df_final


# In[7]:


df_final.groupby('score').count()


# In[8]:


model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


# In[9]:


tokenized = df_final['review'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


# In[10]:


max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])


# In[11]:


np.array(padded).shape


# In[12]:


attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape


# In[13]:


input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)


# In[14]:


features = last_hidden_states[0][:,0,:].numpy()


# In[15]:


labels = df_final['score']


# In[16]:


train_features, test_features, train_labels, test_labels = train_test_split(features, labels)


# In[17]:


lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)


# In[18]:


lr_clf.score(test_features, test_labels)


# In[19]:


from sklearn.dummy import DummyClassifier
clf = DummyClassifier()

scores = cross_val_score(clf, train_features, train_labels)
print("Dummy classifier score: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:





# In[ ]:




