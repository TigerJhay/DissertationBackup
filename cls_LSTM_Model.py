import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re
import json
import mysql.connector
from sqlalchemy import create_engine

# NLTK downloads
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

mysqlconn = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="dbmain_dissertation"
)
sqlengine = create_engine('mysql+mysqlconnector://root@localhost/dbmain_dissertation', pool_recycle=1800)

sqlstring_cm = "SELECT Reviews,Rating, Model FROM gadget_reviews"
temp_df_cm = pd.read_sql(sqlstring_cm, mysqlconn)
df = temp_df_cm.dropna(subset=['Reviews'])

df = df[df['Reviews'].str.strip() != '']
df = pd.DataFrame(temp_df_cm)

# Label: Recommend if Rating ≥ 4, Not Recommend if ≤ 2
df = df[df['Rating'] != 3]
df['Label'] = df['Rating'].apply(lambda r: 1 if r >= 4 else 0)

# Text Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return tokens

# Build vocabulary
df = df.dropna(subset=['Model', 'Reviews'])  # Ensure no None values

all_tokens = [
    token
    for _, row in df.iterrows()
    for token in preprocess(str(row['Model']) + " " + str(row['Reviews']))
]


vocab = {word: idx + 1 for idx, word in enumerate(set(all_tokens))}  # 0 reserved for padding
vocab["<PAD>"] = 0

with open('vocab.json', 'w') as f:
    json.dump(vocab, f)

MAX_LEN = 30

def encode(text):
    tokens = preprocess(text)
    ids = [vocab.get(t, 0) for t in tokens][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))

# Custom Dataset
class ReviewDataset(Dataset):
    def __init__(self, df):
        self.texts = [encode(row['Model'] + " " + row['Reviews']) for _, row in df.iterrows()]
        self.labels = df['Label'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

# Train/Test split
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)
train_ds = ReviewDataset(train_df)
test_ds = ReviewDataset(test_df)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1)

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.squeeze(0)
        out = torch.sigmoid(self.fc(h_n)).view(-1)
        return out

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(len(vocab), embed_dim=32, hidden_dim=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop
print("Training model...\n")
for epoch in range(5):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "lstm_model2.pt")

# Inference function
def predict_product(product_name):
    related_reviews = df[df['Model'].str.lower() == product_name.lower()]
    if related_reviews.empty:
        return "Product not found."

    model.eval()
    predictions = []
    with torch.no_grad():
        for _, row in related_reviews.iterrows():
            input_tensor = torch.tensor([encode(row['Model'] + " " + row['Reviews'])], dtype=torch.long).to(device)
            rating_tensor = torch.tensor([row['Rating']], dtype=torch.float).to(device)  # Add rating
            output = model(input_tensor, rating_tensor).item()
            predictions.append(output)
    avg_pred = np.mean(predictions)
    return "Recommend" if avg_pred >= 0.5 else "Not Recommend"

# User input for prediction
user_input = input("\nEnter Gadget name: ").strip()
result = predict_product(user_input)
print(f"\nPrediction for '{user_input}': {result}")
