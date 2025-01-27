import pandas as pd
import numpy as np
from numpy import array
import re
import nltk 

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


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

from googletrans import Translator
translater = Translator()

#Access and load the dataset record of reviews
#df_reviews = pd.read_csv("./templates/Amazon_Review.csv")
df_reviews = pd.read_csv("./templates/Amazon_Review.csv")
df_reviews.head(20)

df_reviews['Reviews'] = df_reviews['Reviews'].str.lower()

df_reviews["Reviews"] = df_reviews["Reviews"].astype(str)
df_reviews["Translated_text"] = df_reviews["Reviews"].apply(translater.translate, src='auto', dest='en').apply(getattr,args=('text',))
df_reviews
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
vectorize = TfidfVectorizer(stop_words='english')
vectorized_value = vectorize.fit_transform(df_reviews["Reviews"])

k_value = 10
k_model = KMeans(n_clusters=k_value, init='k-means++', max_iter=100, n_init=1)
k_model.fit(vectorized_value)

df_reviews["clusters"] = k_model.labels_
df_reviews.head()


cluster_groupby = df_reviews.groupby("clusters")

for cluster in cluster_groupby.groups:
    f = open("cluster"+str(cluster)+".csv","w")
    data = cluster_groupby.get_group(cluster)[["Rating", "Reviews"]]
    f.write(data.to_csv(index_label="id"))
    f.close()

center_gravity = k_model.cluster_centers_.argsort()[:,::-1]
terms = vectorize.get_feature_names_out()

for ctr in range(k_value):
    print ("Cluster %d: " % ctr)
    for ctr2 in center_gravity[ctr, :10]:
        print ("%s" % terms[ctr2])
    print ("---------------------")