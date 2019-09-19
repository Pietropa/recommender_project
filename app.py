# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:15:11 2019

@author: admin
"""

from flask import Flask
from flask import render_template, request
from recommender import get_ml_recommendations, retrain_nmf

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('bootstrap.html')

@app.route('/recommender')
def show_recommender():
    user_input = list(request.args.to_dict())
    print(user_input)
    results = []
    for each in user_input:
        results.append((each,5))
    #recommendation = get_ml_recommendations(results)
    #return render_template('data.html', data=recommendation)
    try:
        recommendation = get_ml_recommendations(results)
    except(IndexError):
        return render_template('error.html')
    return render_template('data.html', data=recommendation)


@app.route('/train')
def train_nmf():
    retrain_nmf()
    return render_template('retrain.html')


#