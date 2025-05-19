import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import os

import pandas as pd
import numpy as np
import mysql.connector
from sqlalchemy import create_engine
import sqlalchemy as sqlalch
mysqlconn = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="dbmain_dissertation"
)
def sub_datacleaning(temp_df):
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

    temp_df["Rating"] = temp_df["Rating"].astype(str)
    temp_df["Rating"] = temp_df["Rating"].astype(int)
    temp_df = temp_df.drop(temp_df[temp_df["Rating"]==3].index, inplace=False)
    return temp_df

sqlengine = create_engine('mysql+mysqlconnector://root@localhost/dbmain_dissertation', pool_recycle=1800)
sqlstring_cm = "SELECT Model, Reviews, Rating FROM gadget_reviews"
temp_df_cm = pd.read_sql(sqlstring_cm, mysqlconn)
temp_df_cm = sub_datacleaning(temp_df_cm)

# Sample Data (replace with your actual data loading)
data = [
    ("Laptop", "Great performance and battery life!", 5),
    ("Mouse", "Works well, comfortable to use.", 4),
    ("Keyboard", "Keys feel a bit mushy.", 2),
    ("Monitor", "Excellent display quality.", 5),
    ("Webcam", "Grainy image, not recommended.", 1),
    ("Laptop", "Fast processor, good for gaming.", 5),
    ("Mouse", "Scroll wheel broke after a month.", 1),
    ("Keyboard", "Love the clicky keys!", 5),
    ("Monitor", "Colors are vibrant and accurate.", 4),
    ("Webcam", "Clear video calls.", 4),
    ("Tablet", "Good for reading and browsing.", 4),
    ("Charger", "Works as expected.", 3),
    ("Headphones", "Amazing sound quality!", 5),
    ("Tablet", "A bit slow.", 2),
    ("Charger", "Stopped working after a few weeks.", 1),
    ("Headphones", "Uncomfortable after long use.", 2),
    ("Laptop", "Excellent build quality", 5),
    ("Laptop", "Poor customer service", 2),
    ("Mouse", "Ergonomic design", 4),
    ("Mouse", "Cheap plastic", 1),
    ("Keyboard", "Backlit keys are great", 5),
    ("Keyboard", "Some keys stopped working", 1),
    ("Monitor", "High refresh rate", 5),
    ("Monitor", "Flickering issues", 2),
    ("Webcam", "Good low-light performance", 4),
    ("Webcam", "Blurry image", 1),
]

# Separate product names, reviews, and ratings
product_names = temp_df_cm['Model']
reviews = temp_df_cm['Reviews']
ratings = temp_df_cm['Ratings']

# Define a threshold for recommendation (e.g., rating >= 4)
recommendation_threshold = 4
recommendations = ["Recommended" if rating >= recommendation_threshold else "Not Recommended" for rating in ratings]

# 1. Data Preprocessing
def build_vocabulary(texts):
    token_counts = Counter()
    for text in texts:
        token_counts.update(text.lower().split())
    vocabulary = {token: i + 2 for i, token in enumerate(token_counts)}
    vocabulary['<PAD>'] = 0
    vocabulary['<UNK>'] = 1
    return vocabulary

def text_to_indices(text, vocabulary):
    return [vocabulary.get(token.lower(), vocabulary['<UNK>']) for token in text.lower().split()]

vocabulary = build_vocabulary(reviews)
indexed_reviews = [text_to_indices(review, vocabulary) for review in reviews]
indexed_products = [product.lower() for product in product_names]

# Create a mapping for product names to numerical indices
unique_products = list(set(indexed_products))
product_to_index = {product: i for i, product in enumerate(unique_products)}
indexed_products_numeric = [product_to_index[product] for product in indexed_products]


# 2. Dataset Class
class ReviewDataset(Dataset):
    def __init__(self, reviews, products, recommendations, vocabulary):
        self.reviews = reviews
        self.products = products
        self.recommendations = recommendations
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = torch.tensor(self.reviews[idx], dtype=torch.long)
        product = torch.tensor(self.products[idx], dtype=torch.long)
        recommendation = 1 if self.recommendations[idx] == "Recommended" else 0
        recommendation = torch.tensor(recommendation, dtype=torch.float32).unsqueeze(0)
        return review, product, recommendation

def pad_sequences(sequences, max_len):
    padded_sequences = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        length = len(seq)
        padded_sequences[i, :length] = torch.tensor(seq, dtype=torch.long)
    return padded_sequences

# Determine the maximum sequence length
max_len = max(len(seq) for seq in indexed_reviews)
padded_reviews = pad_sequences(indexed_reviews, max_len)

# Split data into training and testing sets
train_reviews, test_reviews, train_products, test_products, train_recommendations, test_recommendations = train_test_split(
    padded_reviews, indexed_products_numeric, recommendations, test_size=0.2, random_state=42)

# Create Datasets and DataLoaders
train_dataset = ReviewDataset(train_reviews, train_products, train_recommendations, vocabulary)
test_dataset = ReviewDataset(test_reviews, test_products, test_recommendations, vocabulary)

batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. LSTM Model
class RecommendationLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_products, product_embedding_dim):
        super(RecommendationLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim + product_embedding_dim, hidden_dim, batch_first=True)
        self.product_embedding = nn.Embedding(num_products, product_embedding_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, review, product):
        embedded_review = self.embedding(review)
        embedded_product = self.product_embedding(product).unsqueeze(1).expand(-1, embedded_review.size(1), -1)
        combined_input = torch.cat((embedded_review, embedded_product), dim=2)
        out, _ = self.lstm(combined_input)
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# Model parameters
embedding_dim = 100
hidden_dim = 128
product_embedding_dim = 50
vocab_size = len(vocabulary)
num_products = len(unique_products)

model = RecommendationLSTM(vocab_size, embedding_dim, hidden_dim, num_products, product_embedding_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_train_predictions = 0
    total_train_samples = 0
    for reviews, products, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(reviews, products)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        predicted_train = (outputs > 0.5).float()
        total_train_samples += labels.size(0)
        correct_train_predictions += (predicted_train == labels).sum().item()

    train_accuracy = correct_train_predictions / total_train_samples
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")


# 5. Evaluation
model.eval()
correct_predictions = 0
total_samples = 0
test_loss = 0  # To accumulate loss over the test set
with torch.no_grad():
    for reviews, products, labels in test_loader:
        outputs = model(reviews, products)
        loss = criterion(outputs, labels) # Calculate loss
        test_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

accuracy = correct_predictions / total_samples
avg_test_loss = test_loss / len(test_loader) # Average test loss
print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

# 6. Save the trained model
def save_model(model, model_path="recommendation_model.pth"):
    """Saves the trained model to a specified path."""
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved to {model_path}")

# Save the model after training
save_model(model)

# 7. Load the trained model
def load_model(model_class, model_path, vocab_size, embedding_dim, hidden_dim, num_products, product_embedding_dim):
    """Loads the trained model from a specified path.

    Args:
        model_class: The class of the model (e.g., RecommendationLSTM).
        model_path: The path to the saved model file.
        vocab_size: The size of the vocabulary.
        embedding_dim: The embedding dimension.
        hidden_dim: The hidden dimension of the LSTM.
        num_products: The number of unique products.
        product_embedding_dim: The product embedding dimension.

    Returns:
        The loaded model.
    """
    model = model_class(vocab_size, embedding_dim, hidden_dim, num_products, product_embedding_dim)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded trained model from {model_path}")
        return model
    else:
        print(f"Model file not found at {model_path}.  Using a new, untrained model.")
        return model
# 8. User Input and Prediction
def predict_recommendation(product_name, model, product_to_index):
    """Predicts the recommendation for a given product name.

    Args:
        product_name: The name of the product to predict for.
        model: The trained model.
        product_to_index: The mapping from product names to indices.

    Returns:
        "Recommended" or "Not Recommended" based on the model's prediction.  Returns "Product Not Found"
        if the product name is not in the training data.
    """
    model.eval()
    with torch.no_grad():
        product_lower = product_name.lower()
        if product_lower not in product_to_index:
            return "Product Not Found"  # Handle unknown product names
        product_index = torch.tensor([product_to_index[product_lower]], dtype=torch.long)

        # Create a dummy review (we only care about the product in this version)
        # The model expects a review, even if we're ignoring it.  Use padding.
        dummy_review = torch.zeros((1, max_len), dtype=torch.long)

        output = model(dummy_review, product_index)
        prediction = (output > 0.5).item()
        return "Recommended" if prediction == 1 else "Not Recommended"

if __name__ == "__main__":
    # Load the model.  Make sure the parameters match how it was trained.
    loaded_model = load_model(RecommendationLSTM, "recommendation_model.pth", vocab_size, embedding_dim, hidden_dim, num_products, product_embedding_dim)

    user_product = input("Enter the product name: ")
    prediction = predict_recommendation(user_product, loaded_model, product_to_index)
    if prediction == "Product Not Found":
        print(f"Product '{user_product}' not found in the training data.")
    else:
        print(f"Based on the trained model, the product '{user_product}' is: {prediction}")
