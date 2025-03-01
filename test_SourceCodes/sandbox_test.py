import pandas as pd
import numpy as np
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
import matplotlib
from matplotlib.dates import MonthLocator, DateFormatter, YearLocator
import mysql.connector
from sqlalchemy import create_engine

mysqldb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="dbmain_dissertation"
)
dbcmd = mysqldb.cursor()
dbcmd.execute("SELECT * FROM gadget_reviews")
myresult = dbcmd.fetchall()

engine = create_engine('mysql+mysqlconnector://root@localhost/dbmain_dissertation', echo=False)
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
custom_stopwords = ['also', 'dad', 'mom', 'kids', 'christmas', 'hoping']

df_reviews = pd.read_sql("SELECT * FROM gadget_reviews", mysqldb)

df_reviews['Reviews'] = df_reviews['Reviews'].str.lower()

if df_reviews["Username"].isnull().values.any() == True:
  df_reviews["Username"] = df_reviews["Username"].fillna("No Username")       
if df_reviews["Date"].isnull().values.any() == True:
  df_reviews["Date"] = df_reviews["Date"].fillna("1/1/11")
if df_reviews["Reviews"].isnull().values.any() == True:
  df_reviews = df_reviews.dropna(subset=['Reviews'], axis=0,how='any',inplace=False)

df_reviews.drop(['Username'],axis='columns',inplace=True)
df_reviews["Reviews"] = df_reviews["Reviews"].str.replace("\n",' ')
df_reviews["Reviews"] = df_reviews["Reviews"].str.replace("\r",' ')
df_reviews = df_reviews.replace(r'http\S+', '', regex=True)
df_reviews = df_reviews.replace(r"x000D", '', regex=True)
df_reviews = df_reviews.replace(r'<[^>]+>', '', regex= True)
df_reviews = df_reviews.replace('[^a-zA-Z0-9]', ' ', regex=True)
df_reviews = df_reviews.replace(r"\s+[a-zA-Z]\s+", ' ', regex=True)
df_reviews = df_reviews.replace(r" +", ' ', regex=True)
df_reviews = df_reviews.replace(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*','', regex=True)
df_reviews = df_reviews.replace(r'\b(' + r'|'.join(custom_stopwords) + r')\b\s*','', regex=True)

def lemmatize_review(review_text):
    words = nltk.word_tokenize(review_text)
    lemmatize_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatize_text = ' '.join(lemmatize_words)
    return lemmatize_text
df_reviews['Reviews'] = df_reviews['Reviews'].apply(lemmatize_review)
df_reviews["Rating"] = df_reviews["Rating"].astype(str)
df_reviews["Rating"] = df_reviews["Rating"].str.replace('[1-2]', '0', regex=True)
df_reviews["Rating"] = df_reviews["Rating"].str.replace('[4-5]', '1', regex=True)
df_reviews = df_reviews.drop(df_reviews[df_reviews["Rating"]=='3'].index, inplace=False)

# ---------------------------------------------------------------------
# T E S T   A R E A
# Data are already clean, next lines will be your test source code
# ---------------------------------------------------------------------

from googletrans import Translator

translator = Translator()
str_to_translate = "Ang daling ikabit at zero so walang galawan at hindi prone magasgas ang mismong camera lenses.  na sulit kasi ang mura din"
translated_string = translator.translate(str_to_translate, src="auto", dest="en")
print ("Converted Value:" + translated_string.text)