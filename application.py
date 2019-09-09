import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import re
import flask
import praw
from flask import Flask, render_template, request

app=Flask(__name__)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,_;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

flairs={1:"AskIndia",
2:"Non-Political",
3:"[R]eddiquette",
4:"Scheduled",
5:"Photography",
6:"Science/Technology",
7:"Politics",
8:"Business/Finance",
9:"Policy/Economy",
10:"Sports",
11:"Food",
12:"AMA"
}

@app.route('/')

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/statistics')
def statistics():
    return flask.render_template('statistics.html')

@app.route("/register", methods=["POST"])
def register():
    if request.method=='POST':
        nm = request.form.get("url")
        mm=nm

        df1=pd.read_csv("https://raw.githubusercontent.com/abhisheksaxena1998/detector--redditv7/master/cleaned_reddit_alphabetav5.csv")
        df1.dropna(inplace=True)
        df1.columns=['index','combined','flair']  
        print (df1.head())

        X_train, X_test, y_train, y_test = train_test_split(df1['combined'],df1['flair'],random_state=0)

        # Fit the CountVectorizer to the training data
        vect = CountVectorizer().fit(X_train)
        X_train_vectorized = vect.transform(X_train)
        model = RandomForestClassifier(n_estimators=500, criterion='entropy')
        model.fit(X_train_vectorized, y_train)
        reddit = praw.Reddit(client_id='WBTxS7rybznf7Q', client_secret='vJUTUflXITBsQMxeviOfG8mCZoA', user_agent='projectreddit', username='Mysterious_abhE', password='Saxena0705')
        #url="https://www.reddit.com/r/MapPorn/comments/a3p0uq/an_image_of_gps_tracking_of_multiple_wolves_in/"
        submission = reddit.submission(url=nm)
        #print (submission.comments[0])
        #print (submission.title)
        submission.comments.replace_more(limit=0)
        #co=[]
        tr=[]
        c=''
        for top_level_comment in submission.comments:        
            c+=top_level_comment.body  


        tr=submission.title+nm+c
        #processed_tweets=[]
        #for tweet in range(len(tr)):  
        processed_tweet = re.sub(r'\W', ' ', str(tr))

            
        # Remove all the special characters
    
        processed_tweet = re.sub(r'http\S+', ' ', processed_tweet)
    
        #processed_tweet = re.sub(r'https?:\/\/+', ' ', processed_tweet)
    
        #processed_tweet=re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ',processed_tweet)
    
        processed_tweet=re.sub(r'www\S+', ' ', processed_tweet)
    
        processed_tweet=re.sub(r'co \S+', ' ', processed_tweet)
        # remove all single characters
        processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
        # Remove single characters from the start
        processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 
 
        # Substituting multiple spaces with single space
        processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
 
        # Removing prefixed 'b'
        processed_tweet = re.sub(r'^b\s+', ' ', processed_tweet)
    
        processed_tweet = re.sub(r'\d','',processed_tweet)
    
        processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)

 
        # Converting to Lowercase
        processed_tweet = processed_tweet.lower()
        processed_tweet=processed_tweet.replace('_',' ')
    
        #processed_tweets.append(processed_tweet)
    
        print ((processed_tweet))            #print(model.predict(vect.transform([tr])))
        filename='combined_modelv5_updated.pkl'

        #pickle.dump(model, open(filename, 'wb'))
        load_lr_model =pickle.load(open(filename, 'rb'))
        #print (load_lr_model.predict(vect.transform([tr])))


    return flask.render_template('result.html',prediction=flairs[int(model.predict(vect.transform([processed_tweet])))],url=mm)
        
