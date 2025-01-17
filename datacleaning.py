import pandas as pd
import numpy as np
from numpy import array
import re
import nltk 
from wordcloud import wordcloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score

# stopwords1 = set(STOPWORDS)
# new_words = ['ref','referee']
# new_stopwords = stopwords.union(new_words)

#custom stopwords, words that are not on nltk.stopwords. These words are not essential in reviews of gadgets.
custom_stopwords = ['also', 'dad', 'mom', 'kids', 'christmas', 'hoping']

nltk.download('stopwords')
nltk.download('wordnet')
#Access and load the dataset record of reviews
df_reviews = pd.read_csv("./templates/Amazon_Review.csv")
df_reviews.head(20)

df_reviews['Reviews'] = df_reviews['Reviews'].str.lower()

# Checking for missing values. Fill necessary and drop if reviews are null
if df_reviews["Username"].isnull().values.any() == True:
    df_reviews["Username"] = df_reviews["Username"].fillna("No Username")       
if df_reviews["Date"].isnull().values.any() == True:
    df_reviews["Date"] = df_reviews["Date"].fillna("1/1/11")
if df_reviews["Reviews"].isnull().values.any() == True:
    df_reviews = df_reviews.dropna(subset=['Reviews'], axis=0,how='any',inplace=False)

df_reviews.drop(['Username','Date'],axis='columns',inplace=True)
#df_reviews.head(20)
# df_reviews
df_reviews["Reviews"] = df_reviews["Reviews"].str.replace("\n",' ')
df_reviews["Reviews"] = df_reviews["Reviews"].str.replace("\r",' ')

#Removal of URL and Links inside of reviews column
df_reviews = df_reviews.replace(r'http\S+', '', regex=True)
df_reviews = df_reviews.replace(r"x000D", '', regex=True)
#df_reviews["Reviews"][400]

#html tag removal
df_reviews = df_reviews.replace(r'<[^>]+>', '', regex= True)
#tag_rem = re.compile(r'<[^>]+>')
#df_reviews = tag_rem.sub('', df_reviews["Reviews"])

#punctuation and character removal
df_reviews = df_reviews.replace('[^a-zA-Z0-9]', ' ', regex=True)
#testvalue = re.sub('[^a-zA-Z0-9]',' ', testvalue)

#Single Character Removal
df_reviews = df_reviews.replace(r"\s+[a-zA-Z]\s+", ' ', regex=True)

#Multiple Spaces Removal
df_reviews = df_reviews.replace(r" +", ' ', regex=True)

#Stopword Removal
df_reviews = df_reviews.replace(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*','', regex=True)
df_reviews = df_reviews.replace(r'\b(' + r'|'.join(custom_stopwords) + r')\b\s*','', regex=True)

#Lemmatize, do I still need it???
df_reviews["Reviews"][418] = lemmatizer.lemmatize(df_reviews["Reviews"][418])
lemmatizer.lemmatize("sets")
df_reviews

#----------------------------------------------------------
#This portion is part of Naive Bayes, Multinomial Algorithm
#----------------------------------------------------------
vectorize = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopwords.words('english'))

y_val = df_reviews['Rating']

#fitting and transform
x_val = vectorize.fit_transform(df_reviews['Reviews'])
x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, random_state=42)
classifier = naive_bayes.MultinomialNB()
classifier.fit(x_train, y_train)
df_reviews
roc_auc_score(y_test, classifier.predict_proba(x_test)[:,1],multi_class='ovo')

gadget_review_array = np.array(["This is a bad review"])
gadget_review_vector = vectorize.transform(gadget_review_array)
classifier.predict(gadget_review_vector)
