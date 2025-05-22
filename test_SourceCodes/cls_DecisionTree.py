import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # For saving and loading the model
mysqlconn = mysql.connector.connect(host="localhost", user="root", password="", database="dbmain_dissertation")
sqlengine = create_engine('mysql+mysqlconnector://root@localhost/dbmain_dissertation', pool_recycle=1800)

def create_dataset():
    """
    Creates a sample dataset for gadget information.

    Returns:
        pandas.DataFrame: A DataFrame containing gadget data.
    """
    data = {
        'Gadget_Brand': ['Apple', 'Samsung', 'Apple', 'Google', 'Samsung', 'OnePlus', 'Google', 'Apple', 'Samsung', 'OnePlus', 'Xiaomi', 'Xiaomi'],
        'Gadget_Type': ['Phone', 'Phone', 'Tablet', 'Phone', 'Phone', 'Phone', 'Phone', 'Phone', 'Tablet', 'Phone', 'Phone', 'Phone'],
        'Gadget_Model': ['iPhone 13', 'Galaxy S21', 'iPad Air', 'Pixel 6', 'Galaxy S21 FE', 'Nord 2', 'Pixel 7', 'iPhone 14', 'Galaxy Tab S8', 'Nord CE 2', 'Redmi Note 11', 'Poco X4'],
        'Recommended': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No']
    }
    
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    """
    Preprocesses the dataset by encoding categorical features.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
        sklearn.preprocessing.LabelEncoder: LabelEncoder for Gadget_Brand.
        sklearn.preprocessing.LabelEncoder: LabelEncoder for Gadget_Type.
        sklearn.preprocessing.LabelEncoder: LabelEncoder for Gadget_Model.
        sklearn.preprocessing.LabelEncoder: LabelEncoder for Recommended.
    """
    # Create label encoders for each categorical column
    brand_encoder = LabelEncoder()
    type_encoder = LabelEncoder()
    model_encoder = LabelEncoder()
    recommended_encoder = LabelEncoder()

    # Fit and transform the columns
    df['Gadget_Brand'] = brand_encoder.fit_transform(df['Gadget_Brand'])
    df['Gadget_Type'] = type_encoder.fit_transform(df['Gadget_Type'])
    df['Gadget_Model'] = model_encoder.fit_transform(df['Gadget_Model'])
    df['Recommended'] = recommended_encoder.fit_transform(df['Recommended'])

    return df, brand_encoder, type_encoder, model_encoder, recommended_encoder

def train_model(X, y):
    """
    Trains a Decision Tree Classifier model.

    Args:
        X (pandas.DataFrame): The feature matrix.
        y (pandas.Series): The target variable.

    Returns:
        sklearn.tree.DecisionTreeClassifier: The trained Decision Tree model.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=42)  # Added random_state for reproducibility

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model (optional, but good practice)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return model

def predict_recommendation(model, brand_encoder, type_encoder, model_encoder, recommended_encoder, product_name):
    """
    Predicts the recommendation for a given product name.

    Args:
        model (sklearn.tree.DecisionTreeClassifier): The trained Decision Tree model.
        brand_encoder (sklearn.preprocessing.LabelEncoder): LabelEncoder for Gadget_Brand.
        type_encoder (sklearn.preprocessing.LabelEncoder): LabelEncoder for Gadget_Type.
        model_encoder (sklearn.preprocessing.LabelEncoder): LabelEncoder for Gadget_Model.
        recommended_encoder (sklearn.preprocessing.LabelEncoder): LabelEncoder for Recommended.
        product_name (str): The name of the product to predict.

    Returns:
        str: The recommendation ("Recommended" or "Not Recommended").
    """
    df = create_dataset() # added to use the data.
    # Get the gadget details based on product name.
    gadget_data = df[df['Gadget_Model'].str.contains(product_name, case=False, na=False)] # added na=False

    if gadget_data.empty:
        return "Product not found."

    # Assume the first match is the correct one.
    gadget_brand = gadget_data.iloc[0]['Gadget_Brand']
    gadget_type = gadget_data.iloc[0]['Gadget_Type']
    gadget_model = gadget_data.iloc[0]['Gadget_Model']
    # Encode the input features using the saved encoders
    try:
        brand_encoded = brand_encoder.transform([gadget_brand])[0]
        type_encoded = type_encoder.transform([gadget_type])[0]
        model_encoded = model_encoder.transform([gadget_model])[0]
    except ValueError as e:
        print(f"Error encoding input: {e}")
        return "Invalid product details.  Ensure Brand, Type, and Model are correct."
    # Create a DataFrame for the input features
    input_data = pd.DataFrame({
        'Gadget_Brand': [brand_encoded],
        'Gadget_Type': [type_encoded],
        'Gadget_Model': [model_encoded],
    })

    # Predict the recommendation
    prediction = model.predict(input_data)[0]
    # Decode the prediction
    recommendation = recommended_encoder.inverse_transform([prediction])[0]
    return recommendation

def save_model(model, brand_encoder, type_encoder, model_encoder, recommended_encoder, filename="gadget_recommendation_model.joblib"):
    """
    Saves the trained model and encoders to a file.

    Args:
        model (sklearn.tree.DecisionTreeClassifier): The trained Decision Tree model.
        brand_encoder (sklearn.preprocessing.LabelEncoder): LabelEncoder for Gadget_Brand.
        type_encoder (sklearn.preprocessing.LabelEncoder): LabelEncoder for Gadget_Type.
        model_encoder (sklearn.preprocessing.LabelEncoder): LabelEncoder for Gadget_Model.
        recommended_encoder (sklearn.preprocessing.LabelEncoder): LabelEncoder for Recommended.
        filename (str, optional): The name of the file to save to.
            Defaults to "gadget_recommendation_model.joblib".
    """
    # Save the model and encoders
    joblib.dump({
        'model': model,
        'brand_encoder': brand_encoder,
        'type_encoder': type_encoder,
        'model_encoder': model_encoder,
        'recommended_encoder': recommended_encoder
    }, filename)
    print(f"Model and encoders saved to {filename}")

def load_model(filename="gadget_recommendation_model.joblib"):
    """
    Loads the trained model and encoders from a file.

    Args:
        filename (str, optional): The name of the file to load from.
            Defaults to "gadget_recommendation_model.joblib".

    Returns:
        tuple: The trained Decision Tree model and the label encoders, or None on error.
    """
    try:
        # Load the model and encoders
        loaded_data = joblib.load(filename)
        print(f"Model and encoders loaded from {filename}")
        return loaded_data['model'], loaded_data['brand_encoder'], loaded_data['type_encoder'], loaded_data['model_encoder'], loaded_data['recommended_encoder']
    except FileNotFoundError:
        print(f"Error: File not found - {filename}")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None, None

def main():
    """
    Main function to run the gadget recommendation system.
    """
    # Create and preprocess the dataset
    df = create_dataset()
    df, brand_encoder, type_encoder, model_encoder, recommended_encoder = preprocess_data(df)

    # Prepare the data for training
    X = df.drop('Recommended', axis=1)
    y = df['Recommended']

    # Check if a saved model exists, if not, train and save a new one
    model, loaded_brand_encoder, loaded_type_encoder, loaded_model_encoder, loaded_recommended_encoder = load_model()

    if model is None:
        print("Training new model...")
        model = train_model(X, y)
        save_model(model, brand_encoder, type_encoder, model_encoder, recommended_encoder)
        #Use the newly trained encoders
        brand_encoder = brand_encoder
        type_encoder = type_encoder
        model_encoder = model_encoder
        recommended_encoder = recommended_encoder
    else:
        print("Loaded existing model.")
        #Use the loaded encoders.
        brand_encoder = loaded_brand_encoder
        type_encoder = loaded_type_encoder
        model_encoder = loaded_model_encoder
        recommended_encoder = loaded_recommended_encoder
    # Get user input and predict recommendation
    while True:
        product_name = input("Enter the product name (or 'exit' to quit): ")
        if product_name.lower() == 'exit':
            break

        recommendation = predict_recommendation(model, brand_encoder, type_encoder, model_encoder, recommended_encoder, product_name)
        print(f"Recommendation: {recommendation}")

if __name__ == "__main__":
    main()

