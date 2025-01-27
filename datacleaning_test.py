import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from numpy import array


df_reviews = pd.read_csv("./templates/Amazon_Review.csv")
df_reviews["Username"] = df_reviews["Username"].fillna("NO_VALUE")  
df_reviews.head(20)
# Dataset exploration
    #df_reviews.shape
    #df_reviews.head(5)

# Checking for missing values. Fill necessary and drop if reviews are null
if df_reviews["Username"].isnull().values.any() == True:
    df_reviews["Username"] = df_reviews["Username"].fillna("NO_VALUE")        
if df_reviews["Date"].isnull().values.any() == True:
    df_reviews["Date"] = df_reviews["Date"].fillna("1/1/11")
if df_reviews["Reviews"].isnull().values.any() == True:
    df_reviews = df_reviews["Reviews"].dropna()
    df_reviews.head(20)

movie_reviews["review"][2]
# You can see that our text contains punctuations, brackets, HTML tags and numbers 
# We will preprocess this text in the next section

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    '''Removes HTML tags: replaces anything between opening and closing <> with empty space'''
    return TAG_RE.sub('', text)

import nltk
nltk.download('stopwords')

def preprocess_text(sen):
    '''Cleans text data up, leaving only 2 or more char long non-stepwords composed of A-Z & a-z only
    in lowercase'''
    
    sentence = sen.lower()

    # Remove html tags
    sentence = remove_tags(sentence)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)  # When we remove apostrophe from the word "Mark's", the apostrophe is replaced by an empty space. Hence, we are left with single character "s" that we are removing here.

    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)  # Next, we remove all the single characters and replace it by a space which creates multiple spaces in our text. Finally, we remove the multiple spaces from our text as well.

    # Remove Stopwords
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    sentence = pattern.sub('', sentence)

    return sentence

# Calling preprocessing_text function on movie_reviews
X = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    X.append(preprocess_text(sen))
    
X[2]

# Converting sentiment labels to 0 & 1
y = movie_reviews['sentiment']

y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# The train set will be used to train our deep learning models 
# while test set will be used to evaluate how well our model performs 



import asyncio
from googletrans import Translator
async def main():
  original_text = "Hola, ¿cómo estás?"
  english_translation = await translate_text(original_text)
  print(f"Original text: {original_text}")
  print(f"Translated text: {english_translation}")

# Run the asynchronous code
asyncio.run(main()) 


async def translate_text(text):
  translator = Translator()
  translation_coroutine = translator.translate(text, dest='en') 
  translation = await translation_coroutine 
  return translation.text

translate_text("Pagkain")

translater = Translator()
translation = translater.translate("Masarap", dest="en")
translation