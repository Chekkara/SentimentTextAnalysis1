# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 00:07:10 2020

@author: admin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"C:\Users\admin\Desktop\nlp\twitt.csv",encoding="ISO-8859-1")

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
copy=dataset['SentimentText']
c=[]
for i in range(0,1000):
    SentimentText=re.sub('[^a-zA-Z]',' ',dataset['SentimentText'][i])
    #print(review)
    SentimentText=SentimentText.lower()
    SentimentText=SentimentText.split()
    SentimentText=[word for word in SentimentText if not word in set(stopwords.words('english'))]
    ps=PorterStemmer()
    #print(review)
    SentimentText=[ps.stem(word) for word in SentimentText if not word in set(stopwords.words('english'))]
    SentimentText= ' '.join(SentimentText)
    #print(review)
    c.append(SentimentText)
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(c).toarray()
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

import keras
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(input_dim=1500,init="random _uniform",activation='sigmod',output_dim=1000))
model.add(Dense(output_dim=100,init="random_uniform",activation='sigmod'))
model.add(Dense(output_dim=1,init="random_uniform",activation='sigmod'))
model.compile(optimise='adam',loss='binary_crossentropy',metrics=['accuracy'])    
model.fit(x_train, y_train,epochs=50,batch_size=10)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test, y_pred)


