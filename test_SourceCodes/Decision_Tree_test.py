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
from sklearn.tree import DecisionTreeClassifier
from googletrans import Translator


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
custom_stopwords = ['also', 'dad', 'mom', 'kids', 'christmas', 'hoping']

df_reviews = pd.read_csv("../templates/Datasets/Main_DataSet.csv", encoding="Latin_1")
#df_reviews = pd.read_csv("../templates/Datasets/Main_Dataset.csv", encoding="latin_1")
#df_reviews.head(20)
#distinct_value = df_reviews["Username"].unique()

#trans_interpreter = Translator()
#df_reviews['Reviews'] = df_reviews.apply(lambda x: trans_interpreter.translate(x['Reviews'],src="auto",dest="en").text, axis=1)
#df_reviews

df_reviews['Reviews'] = df_reviews['Reviews'].str.lower()
df_reviews["Reviews"] = df_reviews["Reviews"].astype(str)
# Checking for missing values. Fill necessary and drop if reviewlas are null
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
df_reviews["Rating"] = df_reviews["Rating"].str.replace('[1-2]', 'negative', regex=True)
df_reviews["Rating"] = df_reviews["Rating"].str.replace('[3-5]', 'positive', regex=True)


df_atttribute = df_reviews.loc[df_reviews["Model"].str.contains('Test Phone', regex=False)]
df_atttribute = df_atttribute.loc[df_reviews["Reviews"].str.contains('battery', regex=False)]
df_atttribute["Reviews"] = df_atttribute["Reviews"].str.replace('[0-9]', "", regex=True)
df_atttribute["Reviews"] = df_atttribute["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}battery\b(?:\W+\w+){0,2})')
df_atttribute = df_atttribute.dropna(axis=0, subset=['Reviews'], how='any')
#df_atttribute.to_csv('battery.csv', index=True)



#vectorize = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='unicode')
#vectorize = CountVectorizer()
test = df_reviews["Rating"].unique()
test
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder
from sklearn import metrics
from sklearn import tree
vectorize = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='unicode')
#vectorize = CountVectorizer()

y_val = df_reviews["Rating"]
x_val = vectorize.fit_transform(df_reviews["Reviews"])

#x_val = label_encoder.fit_transform(df_reviews["Reviews"])
x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=0.2, random_state=0)

#use for pruning
#dec_tree = DecisionTreeClassifier(max_depth=5, random_state=0)
dec_tree = DecisionTreeClassifier(ccp_alpha=0.002, random_state=0)
dec_tree.fit(x_train, y_train).tree_.node_count
#dec_tree.fit(x_train, y_train)
y_pred = dec_tree.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
tree.plot_tree(dec_tree)
plt.show()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report   (y_test, y_pred))