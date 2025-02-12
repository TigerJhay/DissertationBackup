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

lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
custom_stopwords = ['also', 'dad', 'mom', 'kids', 'christmas', 'hoping']

df_reviews = pd.read_csv("../templates/Datasets/Main_DataSet.csv", encoding="Latin_1")
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
df_reviews["Rating"] = df_reviews["Rating"].str.replace('[1-2]', 'negative', regex=True)
df_reviews["Rating"] = df_reviews["Rating"].str.replace('[4-5]', 'positive', regex=True)
df_reviews = df_reviews.drop(df_reviews[df_reviews["Rating"]=='3'].index, inplace=False)

# ---------------------------------------------------------------------
# T E S T   A R E A
# Data are already clean, next lines will be your test source code
# ---------------------------------------------------------------------

#--------------------------------------------------------------------------------------------
#Extracting phrases for creating corpora that will be use in decision tree recommendation
#--------------------------------------------------------------------------------------------

df_reviews = df_reviews.drop(axis=1, columns=["Date"])
df = pd.DataFrame()
def extract_attrib(attrib_value):
  #df_atttribute = df_reviews.loc[df_reviews["Model"].str.contains('Test Phone', regex=False)]
  df_temp = df_reviews.loc[df_reviews["Reviews"].str.contains(attrib_value, regex=False)]
  df_temp["Reviews"] = df_temp["Reviews"].str.replace('[0-9]', "", regex=True)
  
  if attrib_value == "battery":
    df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}battery\b(?:\W+\w+){0,2})')
  elif attrib_value == "speed":
    df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}speed\b(?:\W+\w+){0,2})')
  elif attrib_value == "memory":
    df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}memory\b(?:\W+\w+){0,2})')
  elif attrib_value == "screen":
    df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}screen\b(?:\W+\w+){0,2})')
  else:
    df_temp["Attribute"] = attrib_value
  df_temp = df_temp.dropna(axis=0, subset=["Reviews"], how='any')
  df_temp = df_temp.drop_duplicates(subset="Reviews")
  df_temp["Attribute"] = attrib_value
  return df_temp

list_attrib = ["battery", "screen", "speed", "memory"]
for attrib in list_attrib:
  df = pd.concat([df, extract_attrib(attrib)])
  #print("\n Value attribute is: ", attrib)
  #df.to_csv(attrib+"_All_corpus.csv", index=False)

attrib_matrix = pd.DataFrame(columns=["Model", "Batt_PR","Batt_NR", "Scr_PR", "Scr_NR", "Spd_PR" "Spd_NR", "Mem_PR", "Mem_NR"])

gadget_list = distinct_value = df_reviews["Model"].unique()
def convert_to_matrix(gadget_name, gadget_attrib):
  df_attrib = df.loc[df["Reviews"].str.contains(gadget_name, regex=False)]
  df_model = df_attrib.loc[df_attrib["Model"].str.contains(gadget_attrib, regex=False)]
  rpos = df_model.loc[df_model["Rating"].str.contains("positive", regex=False)].count()
  rneg = df_model.loc[df_model["Rating"].str.contains("negative", regex=False)].count()
  row_value = [gadget_name, rpos, rneg]
  print (row_value)
  return row_value


for gadget_value in gadget_list:
  attrib_matrix["Model"] = gadget_list

attrib_matrix["Batt_PR"] = "Testing"
attrib_matrix.loc[attrib_matrix["Model"].str.contains("iPad 9th Gen")] = "test"

for gadget_value in gadget_list:
  for attrib in list_attrib:
    attrib_matrix["Model"][]
    # print (gadget_value + " " + attrib)
    row_value = convert_to_matrix(gadget_value, attrib)
    
    
    # row_value = pd.concat([attrib_matrix, convert_to_matrix(gadget_value, attrib)])

row_value
#all_corpus = pd.concat([df_atttribute1, df_atttribute2, df_atttribute3, df_atttribute4])
#all_co1rpus.to_csv("All_corpus.csv", index=False)