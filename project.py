#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[ ]:





# In[2]:


from google.colab import drive ## drive a bağlanmak için
drive.mount('/content/drive')


# In[3]:


get_ipython().system("cp /content/drive/MyDrive/project/data.csv /tmp/ ## drive den dosyayı cloab'a kopyalıyoruz. Bu sayede data'yı daha hızlı işleyebiliyoruz.")


# In[4]:


data = pd.read_csv('/tmp/data.csv') ## datayı yüklüyoruz.
data.head() #datasetin ilk 5 verisini gösterir


# In[5]:


data['Haber Gövdesi'] = data['Haber Gövdesi'].apply(lambda x: ' '.join([re.sub(r"[^a-zA-Z]", " ", word.lower()) for word in x.split() ]))
## datadan numaraları ve noktalama işlemlerini çıkarıyoruz. Sadece küçük harf ve büyük harflerden oluşan kelimeler kalıyor.
categories = data['Sınıf'].values
texts = data['Haber Gövdesi'].values


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(texts, categories, random_state=1, test_size = 0.2) # verisetini eğitim ve test diye ayırıyoruz


# In[7]:


vect = CountVectorizer()
vect.fit(X_train) ## vektörize etmek için modeli eğitiyoruz.

X_train_dtm = vect.transform(X_train)  ## train verilerini vektörize ediyoruz. yani veri seti kaç boyutluysa artık, o kadar uzay oluşturup kelimeleri vektörleştiriyoruz.

print(type(X_train_dtm), X_train_dtm.shape)

X_test_dtm = vect.transform(X_test)  ## train verilerini vektörize ediyoruz. yani veri seti kaç boyutluysa artık, o kadar uzay oluşturup kelimeleri vektörleştiriyoruz.
print(type(X_test_dtm), X_test_dtm.shape)


# In[8]:


tfidf_transformer = TfidfTransformer() ## öznitelik çıkarmak için tf-idf yöntemini kullanıyoruz.
tfidf_transformer.fit(X_train_dtm)
tfidf_transformer.transform(X_train_dtm)


# ### Naive Bayes

# In[9]:


nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)


# In[10]:


y_pred_class = nb.predict(X_test_dtm)
print(classification_report(y_test, y_pred_class))


# In[11]:


result = confusion_matrix(y_test, y_pred_class , normalize='pred')
target_names = data.Sınıf.unique()

df_cm = pd.DataFrame(result, index = [i for i in target_names],
                  columns = [i for i in target_names])

plt.figure(figsize = (10,7))
color = sns.light_palette("seagreen", as_cmap=True)
sns.heatmap(df_cm, annot=True, cmap=color)


# ### SGD Classifier

# In[28]:


clf_sgd = SGDClassifier(max_iter=500, tol=1e-3)
clf_sgd.fit(X_train_dtm, y_train)


# In[29]:


y_pred_class = clf_sgd.predict(X_test_dtm)
print(classification_report(y_test, y_pred_class))


# In[30]:


result = confusion_matrix(y_test, y_pred_class , normalize='pred')
target_names = data.Sınıf.unique()

df_cm = pd.DataFrame(result, index = [i for i in target_names],
                  columns = [i for i in target_names])

plt.figure(figsize = (10,7))
color = sns.light_palette("seagreen", as_cmap=True)
sns.heatmap(df_cm, annot=True, cmap=color)


# In[ ]:




