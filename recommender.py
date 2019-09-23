    # -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:21:14 2019
    
 @author: admin
"""
    
import numpy as np
import pandas as pd
import pickle as picklerick
from sklearn.decomposition import NMF
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table

#1 NMF always recommends the same film
#2 sql umr query takes a really long time
#3 sql umr is transposed

engine = create_engine(f'postgres://postgres:test123@localhost/movies2')
base = declarative_base(engine)
Session = sessionmaker(engine)
session = Session()
metadata = base.metadata
ratings = Table('ratings', metadata, autoload=True)
movies = Table('movies', metadata, autoload=True)
tags = Table('tags', metadata, autoload=True)
umr = Table('user_movie_ratings', metadata, autoload=True)

#print(base.metadata.tables['movies'].columns.keys())

    
def retrain_nmf():
    #this is a function which retrains periodically my nmf model
    #it should be trained on the latest user-ratings matrix available
    R = np.array(session.query(umr).all()).T
    model = NMF(n_components=2)
    model.fit(R)
    Q = model.components_
    P = model.transform(R)
    error = model.reconstruction_err_
    nR = np.dot(P, Q)
    #pickle my model
    list_pickle_path = 'nmf.pkl'
    nmf_pickle = open(list_pickle_path, 'wb')
    picklerick.dump(model, nmf_pickle)
    nmf_pickle.close()
    return


def get_ml_recommendations(results):
    #load an nmf model
    list_pickle_path = 'nmf.pkl'
    nmf_unpickle = open(list_pickle_path, 'rb') #rb = ready binary
    model = picklerick.load(nmf_unpickle)

    #find out the movie_id for each movie title
    movie_titles = [x[0] for x in results]
    movie_ratings = [x[1] for x in results]
    movie_ids = []
    for title in movie_titles:
        #SELECT MOVIES.MOVIESID FROM MOVIES WHERE MOVIES.TITLE LIKE '%TITLE%' LIMIT 1;
        db_result = session.query(movies).filter(movies.columns.title.ilike(f'%{title}%')).limit(1).all()
        print(db_result)
        #"{}".format(title)
        
        #'GIVE ME MOVIE NAME WITH MOVIENAME:APPEND SAME LOOP BUT DIFFERENT COLUMN OUTCOME
        
        movie_ids.append(db_result[0][0]) 

    #create an array of len == no. of columns in umr, works because umr is transposed
    data_len = session.query(umr).count()
    query = np.full(data_len, 3.5) #fill it with median rating

    #except for relevant film titles, which are filled with the rating
    for i in range(len(movie_ids)):
        query[movie_ids[i]] = movie_ratings[i]

    query = query.reshape(-1,1).T

#added, where do I find names?     
   # for i in range(len(movie_ids)):
       # query[movie_names[i]] = movie_ratings[i]
    
        
     #NMF MODEL SUBMATRICIES   
    Q = model.components_
    #in this case, a new user providing ratings for the 3 movies.
    P = model.transform(query) #user matrix
    recommendation = np.dot(P,Q)[0] #take the result of the prediction
    
    for each in movie_ids:
        each = 0
    recommendation.append(each)
    
    return recommendation
    #here I insert the exclusion rule
    recommendation = np.argsort(recommendation) #get the index of the best values, why variable is the same?
    
    random = recommendation[-5:][np.random.randint(0,4)] # pick one at random
    #.filter(movies.columns.title != f'%{title}%') added as a test for filtering same title
    film = session.query(movies.columns.title).filter(movies.columns.index == f'{random}').limit(1).all()[0][0] #.title(f'{title}')
    return str(film)