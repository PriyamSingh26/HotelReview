# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 22:50:24 2020

@author: Adiba
"""

import pandas as pd 
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from nltk.util import ngrams
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import nltk
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from imblearn.over_sampling import SMOTE

app = Flask(__name__,template_folder='C:/Users/Agnelo Christy/Desktop/templates')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    df = pd.read_csv('train.csv')
    df['Rating'] = df['Rating'].replace(2,1)
    df['Rating'] = df['Rating'].replace(3,2)
    df['Rating'] = df['Rating'].replace(4,2)
    df['Rating'] = df['Rating'].replace(5,3)
    X = df['Review']
    y = df['Rating']
        
    tfidf = TfidfVectorizer(ngram_range = (1,5))
    X_vect = tfidf.fit_transform(X)
                
    oversample = SMOTE()
    X_bal, y_bal = oversample.fit_resample(X_vect,y)
        
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.33, random_state=42)
                
    global classSVC
    classSVC = LinearSVC()
    classSVC.fit(X_train,y_train)
    classSVC.score(X_test,y_test)
    if request.method == 'POST':
        review = request.form['Review']
        data = [review]
        vect = tfidf.transform(data).toarray()
        my_prediction = classSVC.predict(vect)        
        
            
    #return render_template('result.html',prediction = my_prediction)
    return render_template('index.html', prediction_text='Your Rating is: {}'.format(my_prediction))



if __name__ == "__main__":
    app.run(debug=True)
