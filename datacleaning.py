import pandas as pd
import numpy as np
import re
import nltk 
from nltk.corpus import stopwords
from numpy import array

df = pd.read_excel("D:/Repository_Reviews2.xlsx")
df
#df = pd.read_csv("D:/Repository_Reviews2.csv")
df.shape
df.head(5)

df = df.drop(columns = "BrndNo")
df = df.drop(columns = "GDNo")
df["Product_Review"] = df["Product_Review"].str.replace("\n",' ')
df

nltk.download('stopwords')
df["Product_Review"][0] = "Test"

#URL and links removal
testvalue = re.sub(r'http\S+', '', df["Product_Review"][2])
testvalue = re.sub(r"x000D",'',testvalue)

#html tag removal
tag_rem = re.compile(r'<[^>]+>')
testvalue = tag_rem.sub('', testvalue)

#punctuation and character removal
testvalue = re.sub('[^a-zA-Z0-9]',' ', testvalue)

#Single Character Removal
testvalue = re.sub(r"\s+[a-zA-Z]\s+", ' ', testvalue)

#Multiple Spaces Removal
testvalue = re.sub(r'\s+', ' ', testvalue)

#Stopword Removal
stopword_pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
testvalue = stopword_pattern.sub('', testvalue)
testvalue