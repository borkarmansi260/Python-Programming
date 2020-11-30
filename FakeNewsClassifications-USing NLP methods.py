#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np


# In[6]:


import pandas as pd


# In[7]:


import os


# In[8]:


print(os.getcwd())
print(os.listdir())r'C:\Users\CSUFTitan\Downloads\fake_or_real_news.csv'


# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
from sklearn.linear_model import PassiveAggressiveClassifier

df = pd.read_csv('fake_or_real_news.csv') # Load data into DataFrame
y = df.label
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33,random_state=53)


# In[12]:


count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train.values)
count_test = count_vectorizer.transform(X_test.values)


# In[13]:


# Initialize the `tfidf_vectorizer` 
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) 

# Fit and transform the training data 
tfidf_train = tfidf_vectorizer.fit_transform(X_train) 

# Transform the test set 
tfidf_test = tfidf_vectorizer.transform(X_test)

print(tfidf_test)


# In[14]:


# Get the feature names of `tfidf_vectorizer` 
print(tfidf_vectorizer.get_feature_names()[-10:])
# Get the feature names of `count_vectorizer` 
print(count_vectorizer.get_feature_names()[0:10])


# In[16]:


import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[17]:


clf = MultinomialNB() 
clf.fit(count_train, y_train)
pred = clf.predict(count_test)
score = accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[18]:


clf = MultinomialNB() 
clf.fit(tfidf_train, y_train)
pred = clf.predict(tfidf_test)
score = accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[28]:


from sklearn.feature_extraction.text import HashingVectorizer
# Initialize the hashing vectorizer
hashing_vectorizer = HashingVectorizer(stop_words='english',n_features=5000,alternate_sign=False)

# Fit and transform the training data 
hashing_train = hashing_vectorizer.fit_transform(X_train)

# Transform the test set 
hashing_test = hashing_vectorizer.transform(X_test)

print(hashing_test)


# In[29]:


from sklearn.feature_extraction.text import HashingVectorizer
# Initialize the hashing vectorizer
hashing_vectorizer = HashingVectorizer(stop_words='english',n_features=5000, alternate_sign=False)

# Fit and transform the training data 
hashing_train = hashing_vectorizer.fit_transform(X_train)

# Transform the test set 
hashing_test = hashing_vectorizer.transform(X_test)

print(hashing_test)


# In[ ]:




