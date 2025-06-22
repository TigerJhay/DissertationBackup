from flask import Flask, session, render_template, request
import pandas as pd 
import numpy as np
from numpy import array
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

lemmatizer = WordNetLemmatizer()
import mysql.connector
from sqlalchemy import create_engine
import sqlalchemy as sqlalch
import gc
import torch
import torch.nn as nn
import joblib 
import json

gc.collect()
# nltk.download('wordnet')
app = Flask(__name__)
mysqlconn = mysql.connector.connect(host="localhost", user="root", password="", database="dbmain_dissertation")
sqlengine = create_engine('mysql+mysqlconnector://root@localhost/dbmain_dissertation', pool_recycle=1800)
      
def sub_datacleaning(temp_df):
        # custom_stopwords = ['also', 'dad', 'mom', 'kids', 'christmas', 'hoping']

        #Remove Column Username since this column is unnecessary
        temp_df["Reviews"] = temp_df["Reviews"].str.lower()
        
        # All records with not value for REVIEWS will be dropped
        if temp_df["Reviews"].isnull().values.any() == True:
            temp_df = temp_df.dropna(subset=['Reviews'], axis=0,how='any',inplace=False)

        # Replace all special characters into black spaces which will also be remove
        temp_df["Reviews"] = temp_df["Reviews"].str.replace("\n",' ')
        temp_df["Reviews"] = temp_df["Reviews"].str.replace("\r",' ')
        temp_df["Reviews"] = temp_df["Reviews"].replace(r'http\S+', '', regex=True)
        temp_df["Reviews"] = temp_df["Reviews"].replace(r"x000D", '', regex=True)
        temp_df["Reviews"] = temp_df["Reviews"].replace(r'<[^>]+>', '', regex= True)        
        temp_df["Reviews"] = temp_df["Reviews"].replace('[^a-zA-Z0-9]', ' ', regex=True)
        temp_df["Reviews"] = temp_df["Reviews"].replace(r"\s+[a-zA-Z]\s+", ' ', regex=True) #Eto
        temp_df["Reviews"] = temp_df["Reviews"].replace(r" +", ' ', regex=True)

        # temp_df_temp = temp_df
        # # Eto nalang
        # temp_df = temp_df_temp

        def tokenize_reviews(review_text):
            review_sentence = word_tokenize(review_text)
            return review_sentence
        temp_df['Reviews'] = temp_df['Reviews'].apply(tokenize_reviews)


        # nltk.download('stopwords')
        # def remove_stopwords(review_text):                  
        #     stop_words = set(stopwords.words('english'))
        #     filtered_text = [word for word in review_text if word not in stop_words]
        #     return filtered_text
        # temp_df['Reviews'] = temp_df['Reviews'].apply(remove_stopwords)


        def lemmatize_review(review_text):
            lemmatizer = WordNetLemmatizer()
            lemmatize_words = [lemmatizer.lemmatize(word) for word in review_text]
            lemmatize_text = ' '.join(lemmatize_words)
            return lemmatize_text
        temp_df['Reviews'] = temp_df['Reviews'].apply(lemmatize_review)
        
        temp_df["Reviews"].replace('', None, inplace=True)
        
        if temp_df["Reviews"].isnull().values.any():
            temp_df = temp_df.dropna(subset=['Reviews'], axis=0,how='any',inplace=False)

        #Rating of the sentiments will be converted into 3 classes
        # 0 - Negative Rating or review, These are with rating of 1 & 2
        # 1 - Positive Rating or review, These are with rating of 4 & 5
        
        temp_df["Rating"] = temp_df["Rating"].astype(str)
        temp_df["Rating"] = temp_df["Rating"].str.replace('[1-2]', '0', regex=True)
        temp_df["Rating"] = temp_df["Rating"].str.replace('[4-5]', '1', regex=True)
        temp_df["Rating"] = temp_df["Rating"].astype(int)
        # 3 - Neutral Rating or review, These are with rating of 3
        # This rating will be drop to be dataframe since these are all neither positive or negative
        temp_df = temp_df.drop(temp_df[temp_df["Rating"]==3].index, inplace=False)
        return temp_df

def sub_decision_tree():

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

    dectree_df = pd.read_sql("SELECT Reviews, Rating FROM gadget_reviews", sqlengine)
    dectree_df = sub_datacleaning(dectree_df)
    X  = dectree_df["Reviews"]

    countvector = CountVectorizer()
    X = countvector.fit_transform(dectree_df["Reviews"])
    y = dectree_df["Rating"].astype("float16")

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=.2)
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    #Convert Values into percent
    # cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix for Decision Tree")
    plt.xticks(np.arange(2)+0.5,["Not Recommended", "Recommended"])
    plt.yticks(np.arange(2)+0.5,["Not Recommended", "Recommended"])
    plt.show()

    # Precision, Recall, F1 Score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Precision: {precision:.3%}")
    print(f"Recall: {recall:.3%}")
    print(f"F1 Score: {f1:.3%}")


sub_decision_tree()