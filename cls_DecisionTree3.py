from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import mysql.connector
from sqlalchemy import create_engine
import sqlalchemy as sqlalch
import joblib  # For saving and loading the model
joblib.__version__
mysqlconn = mysql.connector.connect(host="localhost", user="root", password="", database="dbmain_dissertation")
sqlengine = create_engine('mysql+mysqlconnector://root@localhost/dbmain_dissertation', pool_recycle=1800)

def sub_datacleaning(temp_df):
    # custom_stopwords = ['also', 'dad', 'mom', 'kids', 'christmas', 'hoping']

    #Remove Column Username since this column is unnecessary
    temp_df["Reviews"] = temp_df["Reviews"].str.lower()
    
    # Checking for missing values. Fill necessary and drop if reviews are null
    if temp_df["Username"].isnull().values.any() == True:
        temp_df["Username"] = temp_df["Username"].fillna("No Username")       
    
    # Date with invalid values will be default to 1/1/11, which also not useful :)
    # Date with no values will also be converted, which also not useful :)
    if temp_df["Date"].isnull().values.any() == True:
        temp_df["Date"] = temp_df["Date"].fillna("1/1/11")

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

mysqlconn.reconnect()
dectree_df = pd.read_sql("SELECT Username, Date, Brand, Type, Model, Reviews, Rating FROM gadget_reviews", mysqlconn)
dectree_df = sub_datacleaning(dectree_df)
X = dectree_df["Reviews"]
y = dectree_df["Rating"].astype("float16")

countvector = CountVectorizer()
X = countvector.fit_transform(dectree_df["Reviews"])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.2)
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

#Save Model
joblib.dump(dtc, "DT_Model2.pkl")
joblib.dump(countvector, "DT_Vector2.pkl")

#Load Model
dtc = joblib.load("DT_Model2.pkl")
countvector = joblib.load("DT_Vector2.pkl")

# --- Function to predict recommendation for a given model ---
def predict_recommendation(model_name):
    try:
        # Retrieve reviews for the specified model from the DataFrame
        mysqlconn.reconnect()
        model_reviews_df = pd.read_sql(f"SELECT Reviews FROM gadget_reviews WHERE Model = '{model_name}'", mysqlconn)
        if model_reviews_df.empty:
            return f"No reviews found for model: {model_name}"

        model_reviews = model_reviews_df["Reviews"].tolist()
        # Transform the new reviews using the *same* CountVectorizer
        model_reviews_vectorized = countvector.transform(model_reviews)

        # Predict the ratings for these reviews
        predicted_ratings = dtc.predict(model_reviews_vectorized)

        recommendation_threshold = .5  # Adjust this threshold as needed
        average_rating = predicted_ratings.mean()

        if average_rating >= recommendation_threshold:
            return f"The model '{model_name}' is likely to be recommended (average predicted rating: {average_rating:.2f})."
        else:
            return f"The model '{model_name}' is not likely to be recommended (average predicted rating: {average_rating:.2f})."

    except Exception as e:
        return f"An error occurred: {e}"

# --- Get user input for the gadget model ---
# gadget_model_input = input("Enter the gadget model you want to check: ")

# --- Predict and display the recommendation ---
# recommendation_result = predict_recommendation(gadget_model_input)
# print(recommendation_result)