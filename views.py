from flask import Blueprint, render_template, request, flash, jsonify
import google.generativeai as genai
import os
import pandas as pd
import numpy as np

#------------------------------
# Libraries and imports to use
#------------------------------
from numpy import array
import re
import nltk #from wordcloud import wordcloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, YearLocator
lemmatizer = WordNetLemmatizer()

# from HTMLparser import HTMLParser

views = Blueprint(__name__, "views")

@views.route("/")
def home():
     return render_template("index.html")
     #return render_template("testscript.html")

# @views.route("/generateResult", methods=["GET","POST"])
# def fetchAIdesc():
#      searchstring = str(request.form['txtsearch'])

@views.route("/testnaivealgo", methods=["GET", "POST"])
def naivebayes_algo():
     gadget_search = str(request.form['txtsearch'])

     custom_stopwords = ['also', 'dad', 'mom', 'kids', 'christmas', 'hoping']

     nltk.download('stopwords')
     nltk.download('wordnet')
     nltk.download('punkt_tab')

     df_reviews = pd.read_csv("./templates/Datasets/Main_Dataset_v1.csv", encoding="ISO-8859-1")
     df_reviews.head(20)

     df_reviews['Reviews'] = df_reviews['Reviews'].str.lower()

     # Checking for missing values. Fill necessary and drop if reviews are null
     if df_reviews["Username"].isnull().values.any() == True:
          df_reviews["Username"] = df_reviews["Username"].fillna("No Username")       
     if df_reviews["Date"].isnull().values.any() == True:
          df_reviews["Date"] = df_reviews["Date"].fillna("1/1/11")
     if df_reviews["Reviews"].isnull().values.any() == True:
          df_reviews = df_reviews.dropna(subset=['Reviews'], axis=0,how='any',inplace=False)

     #Remove Column Username since this column is unnecessary
     df_reviews.drop(['Username'],axis='columns',inplace=True)

     #Replace special tags inside sentiment
     df_reviews["Reviews"] = df_reviews["Reviews"].str.replace("\n",' ')
     df_reviews["Reviews"] = df_reviews["Reviews"].str.replace("\r",' ')

     #Removal of URL and Links inside of reviews column
     df_reviews = df_reviews.replace(r'http\S+', '', regex=True)
     df_reviews = df_reviews.replace(r"x000D", '', regex=True)

     #HTML tag removal
     df_reviews = df_reviews.replace(r'<[^>]+>', '', regex= True)

     #Punctuation and character removal
     df_reviews = df_reviews.replace('[^a-zA-Z0-9]', ' ', regex=True)

     #Single Character Removal
     df_reviews = df_reviews.replace(r"\s+[a-zA-Z]\s+", ' ', regex=True)

     #Multiple Spaces Removal
     df_reviews = df_reviews.replace(r" +", ' ', regex=True)

     #Stopword Removal
     df_reviews = df_reviews.replace(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*','', regex=True)
     df_reviews = df_reviews.replace(r'\b(' + r'|'.join(custom_stopwords) + r')\b\s*','', regex=True)

     def lemmatize_review(review_text):
          words = nltk.word_tokenize(review_text)
          lemmatize_words = [lemmatizer.lemmatize(word) for word in words]
          lemmatize_text = ' '.join(lemmatize_words)
          return lemmatize_text
     df_reviews['Reviews'] = df_reviews['Reviews'].apply(lemmatize_review)

     #Rating of the sentiments will be converted into 3 classes
     # 0 - Negative Rating or review, These are with rating of 1 & 2
     # 1 - Positive Rating or review, These are with rating of 4 & 5
     # 3 - Neutral Rating or review, These are with rating of 3
     df_reviews["Rating"] = df_reviews["Rating"].astype(str)
     df_reviews["Rating"] = df_reviews["Rating"].str.replace('[1-2]', '0', regex=True)
     df_reviews["Rating"] = df_reviews["Rating"].str.replace('[3-5]', '1', regex=True)
     #drop yung 3 rating

     df_naivebayes = pd.DataFrame(df_reviews)
     df_lstm = pd.DataFrame(df_reviews)
     df_kmeans = pd.DataFrame(df_reviews)

     #----------------------------------------------------------
     #This portion is part of Naive Bayes, Multinomial Algorithm
     #----------------------------------------------------------
     vectorize = CountVectorizer()

     y_val = df_naivebayes['Rating']
     x_val = df_naivebayes['Reviews']
     x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=0.2, random_state=0)
     x_train_count = vectorize.fit_transform(x_train.values)
     x_train_count.toarray()

     classifier = naive_bayes.MultinomialNB()
     classifier.fit(x_train_count, y_train)
     
     #no.array() should be use with predicttion dataset, values encoded are just for testing of algorithm
     gadget_review_array = np.array([gadget_search])
     gadget_review_vector = vectorize.transform(gadget_review_array)
     nb_result = classifier.predict(gadget_review_vector)
    
     for result in nb_result:
          value = "No value"
          if result==0:
               value = "The sentiment is positive"
          else:
               value = "The sentiment is positive"
     
     flash (value)
     return render_template("index.html")

@views.route("/generateResult", methods=["GET","POST"])
def fetchAIdesc():
     searchstring = str(request.form['txtsearch'])
     genai.configure(api_key="AIzaSyDgRaOiicnXJSx_GNtfvuNxKLhCDCDpHhQ")
     model = genai.GenerativeModel("gemini-1.5-flash")
     response = model.generate_content(searchstring)

     #response.text
     flash("***Updated Generated AI response: " + str(response.text.replace("**","\n")))
     return render_template("index.html")
  
@views.route("/search_gadget_description", methods=["GET","POST"])
def fetchAIdescription():
     searchstring = str(request.form['txtSearchValue'])
     genai.configure(api_key="AIzaSyDgRaOiicnXJSx_GNtfvuNxKLhCDCDpHhQ")
     model = genai.GenerativeModel("gemini-1.5-flash")
     response = model.generate_content(searchstring)    
     flash("Generated AI response: " + str(response.text))
     return render_template("testscript.html")

@views.route("/profile")
def profile():
    args = request.args
    name = args.get('name')
    return render_template("index.html", name=name)