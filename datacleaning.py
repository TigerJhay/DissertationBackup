import pandas as pd
import numpy as np
from numpy import array
import re
import nltk 
#from wordcloud import wordcloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes



# stopwords1 = set(STOPWORDS)
# new_words = ['ref','referee']
# new_stopwords = stopwords.union(new_words)

#custom stopwords, words that are not on nltk.stopwords. These words are not essential in reviews of gadgets.
custom_stopwords = ['also', 'dad', 'mom', 'kids', 'christmas', 'hoping']

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

#Access and load the dataset record of reviews
#df_reviews = pd.read_csv("./templates/Amazon_Review.csv")
df_reviews = pd.read_csv("./templates/TestData_10_rows_only.csv")
df_reviews.head(20)

df_reviews['Reviews'] = df_reviews['Reviews'].str.lower()

# Checking for missing values. Fill necessary and drop if reviews are null
if df_reviews["Username"].isnull().values.any() == True:
    df_reviews["Username"] = df_reviews["Username"].fillna("No Username")       
if df_reviews["Date"].isnull().values.any() == True:
    df_reviews["Date"] = df_reviews["Date"].fillna("1/1/11")
if df_reviews["Reviews"].isnull().values.any() == True:
    df_reviews = df_reviews.dropna(subset=['Reviews'], axis=0,how='any',inplace=False)

df_reviews.drop(['Username'],axis='columns',inplace=True)

#replace special tags inside sentiment
df_reviews["Reviews"] = df_reviews["Reviews"].str.replace("\n",' ')
df_reviews["Reviews"] = df_reviews["Reviews"].str.replace("\r",' ')

#Removal of URL and Links inside of reviews column
df_reviews = df_reviews.replace(r'http\S+', '', regex=True)
df_reviews = df_reviews.replace(r"x000D", '', regex=True)

#html tag removal
df_reviews = df_reviews.replace(r'<[^>]+>', '', regex= True)

#punctuation and character removal
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
df_reviews["Rating"] = df_reviews["Rating"].str.replace('[4-5]', '1', regex=True)

#Vectorize process
vectorize = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii')
#vectorize = CountVectorizer()

#----------------------------------------------------------
#This portion is part of Naive Bayes, Multinomial Algorithm
#----------------------------------------------------------
y_val = df_reviews['Rating']
x_val = df_reviews['Reviews']
x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=0.2, random_state=0)
x_train_count = vectorize.fit_transform(x_train.values)
x_train_count.toarray()

#dfxls = x_train_count.toarray()[:2]
#dfxls = pd.DataFrame(x_train_count.toarray())
#dfxls.to_excel("x_train.xlsx")

classifier = naive_bayes.MultinomialNB()
classifier.fit(x_train_count, y_train)
#roc_auc_score(y_test, classifier.predict_proba(x_test)[:,1],multi_class='ovo')

gadget_review_array = np.array(["Capacity are good"])
gadget_review_vector = vectorize.transform(gadget_review_array)
classifier.predict(gadget_review_vector)