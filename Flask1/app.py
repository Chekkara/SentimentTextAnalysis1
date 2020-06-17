# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:08:31 2020

@author: admin
"""
from flask import Flask,request,render_template
from keras.models import load_model

import numpy as np
global model, graph
import tensorflow as tf
from tensorflow.python.keras.backend
import set_session
sess=tf.Session()
graph =  tf.get_default_graph()
set_session(sess)
model = load_model('shrijani10.h5.py')

app = Flask(__name__)

@app.route('/')#when even the browser finds localhost:5000 then
def home():#excecute this function
    return render_template('index1.html')#this function is returing the index.html file
@app.route('/login', methods =['POST']) #when you click submit on html page it is redirection to this url
def login():#as soon as this url is redirected then call the below functionality
    a = request.form['a']

    d = request.form['s']
    if (d == "good"):
        s1,s2,s3 = 0,0,1
    if (d == "bad"):
        s1,s2,s3 = 0,1,0
    if (d == "normal"):
        s1,s2,s3 = 1,0,0
        
    total = [[s1,s2,s3,a]]
    with graph.as_default():
        ypred = model.predict(np.array(total))
        y = ypred[0][0]
        print(ypred)

    # from html page what ever the text is typed  that is requested from the form functionality and is stored in a name variable
    return render_template('index1.html' ,abc = y)#after typing the name show this name on index.html file where we have created a varibale abc


if __name__ == '__main__':
    app.run(debug = True)