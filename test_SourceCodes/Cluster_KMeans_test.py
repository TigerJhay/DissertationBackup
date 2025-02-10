import pandas as pd
import numpy as np
from numpy import array
import re
import nltk 
import matplotlib.pyplot as plt
import matplotlib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.cluster import KMeans
#from translate import Translator
from googletrans import Translator

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
custom_stopwords = ['also', 'dad', 'mom', 'kids', 'christmas', 'hoping']
#Access and load the dataset record of reviews
#df_reviews = pd.read_csv("./templates/Amazon_Review.csv")
df_reviews = pd.read_csv("../templates/Datasets/Main_Dataset.csv", encoding="latin_1")
df_reviews.head(20)
distinct_value = df_reviews["Model"].unique()
#df_reviews = df_reviews[df_reviews["Reviews"].str.contains("ipad")]
#trans_interpreter = Translator()
#df_reviews.apply(lambda x: trans_interpreter.translate(x['Reviews'],src="auto",dest="en").text, axis=1)
#df_reviews

#translated_string = trans_interpreter.translate("maayos naman ang lagay ko", src="auto", dest="en").text
#df_reviews["Reviews"] =  df_reviews["Reviews"].apply(lambda x: trans_interpreter.translate(x, src="auto", dest="en").text)
#df_reviews["Translated_text"] = df_reviews["Reviews"].apply(trans_interpreter.translate, src='auto', dest='en').text

df_reviews['Reviews'] = df_reviews['Reviews'].str.lower()
df_reviews["Reviews"] = df_reviews["Reviews"].astype(str)
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

#Lemmatize dataframe, do I still need it???
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

df_reviews["Reviews"] = df_reviews["Reviews"].values.astype("U")
#vectorize = TfidfVectorizer(stop_words='english')
vectorize = CountVectorizer()
vectorized_value = vectorize.fit_transform(df_reviews["Reviews"])

k_value = 10
k_model = KMeans(n_clusters=k_value, init='k-means++', max_iter=100, n_init=1)
kmean_model = k_model.fit_transform(vectorized_value)
kmean_model
#df_reviews["clusters"] = k_model.labels_
#df_reviews.head()

center_gravity = k_model.cluster_centers_.argsort()[:,::-1]
terms = vectorize.get_feature_names_out()

for ctr in range(k_value):
    print ("Cluster %d: " % ctr)
    for ctr2 in center_gravity[ctr, :10]:
        print ("%s" % terms[ctr2])
    print ("---------------------")

#plt.scatter(k_model.cluster_centers_)
#plt.xlabel(vectorized_value)
#plt.ylabel(vectorized_value)
#plt.show()