import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import mysql.connector
from sqlalchemy import create_engine
import sqlalchemy as sqlalch

mysqlconn = mysql.connector.connect(host="localhost", user="root", password="", database="dbmain_dissertation")
sqlengine = create_engine('mysql+mysqlconnector://root@localhost/dbmain_dissertation', pool_recycle=1800)
# sqlstring = "SELECT * FROM gadget_reviews where Brand='" +brands+"' and Type='"+type+"' and Model='"+gadgetmodel + "'"
sqlstring_dt = "SELECT Brand, Type, Model, Rating FROM gadget_reviews"
temp_df = pd.read_sql(sqlstring_dt, mysqlconn)


df = pd.DataFrame(temp_df)

# Calculate the average rating for each unique product (Brand, Type, Model)
average_ratings = df.groupby(['Brand', 'Type', 'Model'])['Rating'].mean().reset_index()

# Determine recommendation based on a threshold (e.g., average rating > 3 is positive)
threshold = 3
average_ratings['Recommended'] = average_ratings['Rating'].apply(lambda x: 'Positive' if x > threshold else 'Negative')

# Merge the recommendation back into the original DataFrame
df = pd.merge(df, average_ratings[['Brand', 'Type', 'Model', 'Recommended']],
                 on=['Brand', 'Type', 'Model'], how='left')

# Encode categorical features
le_brand = LabelEncoder()
df['Brand_Encoded'] = le_brand.fit_transform(df['Brand'])

le_type = LabelEncoder()
df['Type_Encoded'] = le_type.fit_transform(df['Type'])

le_model = LabelEncoder()
df['Model_Encoded'] = le_model.fit_transform(df['Model'])

le_recommend = LabelEncoder()
df['Recommended_Encoded'] = le_recommend.fit_transform(df['Recommended'])

# Features (X) and target (y)
X = df[['Brand_Encoded', 'Type_Encoded', 'Model_Encoded']].drop_duplicates()
y = df[['Brand', 'Type', 'Model', 'Recommended_Encoded']].drop_duplicates()['Recommended_Encoded']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Function to predict recommendation for a given product name
def predict_recommendation(product_name, trained_model, brand_encoder, type_encoder, model_encoder, recommend_encoder, threshold):
    try:
        parts = product_name.split()
        if len(parts) != 3:
            return "Invalid input format. Please enter Brand Type Model (e.g., Samsung Phone S21)."

        brand, gadget_type, model_name = parts[0], parts[1], parts[2]

        brand_encoded = brand_encoder.transform([brand])[0]
        type_encoded = type_encoder.transform([gadget_type])[0]
        model_encoded = model_encoder.transform([model_name])[0]

        input_data_encoded = [[brand_encoded, type_encoded, model_encoded]]
        prediction_encoded = trained_model.predict(input_data_encoded)[0]
        prediction = recommend_encoder.inverse_transform([prediction_encoded])[0]

        if prediction == 'Positive':
            return "Recommended"
        else:
            return "Not Recommended"

    except ValueError:
        return "Product not found in the trained data."

# User input for product name
user_product = input("Enter the product name (Brand Type Model): ")
recommendation = predict_recommendation(user_product, model, le_brand, le_type, le_model, le_recommend, threshold)
print(f"Based on the trained model, '{user_product}' is: {recommendation}")

# --- Saving the trained model and encoders ---
model_filename = 'gadget_recommendation_model.joblib'
encoders_filename = 'gadget_recommendation_encoders.joblib'

joblib.dump(model, model_filename)
joblib.dump({'brand_encoder': le_brand, 'type_encoder': le_type, 'model_encoder': le_model, 'recommend_encoder': le_recommend, 'threshold': threshold}, encoders_filename)

print(f"\nTrained model saved as '{model_filename}'")
print(f"Trained encoders and threshold saved as '{encoders_filename}'")

# --- Loading the trained model and encoders ---
def load_model_and_encoders(model_filename, encoders_filename):
    loaded_model = joblib.load(model_filename)
    loaded_data = joblib.load(encoders_filename)
    return loaded_model, loaded_data['brand_encoder'], loaded_data['type_encoder'], loaded_data['model_encoder'], loaded_data['recommend_encoder'], loaded_data['threshold']

loaded_model, loaded_brand_encoder, loaded_type_encoder, loaded_model_encoder, loaded_recommend_encoder, loaded_threshold = load_model_and_encoders(model_filename, encoders_filename)
print("\nTrained model and encoders loaded successfully.")

# Example of using the loaded model
user_product_loaded = input("Enter another product name to check with the loaded model (Brand Type Model): ")
recommendation_loaded = predict_recommendation(user_product_loaded, loaded_model, loaded_brand_encoder, loaded_type_encoder, loaded_model_encoder, loaded_recommend_encoder, loaded_threshold)
print(f"Based on the loaded model, '{user_product_loaded}' is: {recommendation_loaded}")