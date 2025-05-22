import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset,Dataset
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import mysql.connector
from sqlalchemy import create_engine

nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('omw-1.4')

mysqlconn = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="dbmain_dissertation"
)
sqlengine = create_engine('mysql+mysqlconnector://root@localhost/dbmain_dissertation', pool_recycle=1800)

def sub_datacleaning(temp_df):

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

    def tokenize_reviews(review_text):
        review_sentence = word_tokenize(review_text)
        return review_sentence
    temp_df['Reviews'] = temp_df['Reviews'].apply(tokenize_reviews)

    def lemmatize_review(review_text):
        lemmatizer = WordNetLemmatizer()
        lemmatize_words = [lemmatizer.lemmatize(word) for word in review_text]
        lemmatize_text = ' '.join(lemmatize_words)
        return lemmatize_text

    temp_df['Reviews'] = temp_df['Reviews'].apply(lemmatize_review)
    temp_df["Reviews"].replace('', None, inplace=True)

    if temp_df["Reviews"].isnull().values.any():
        temp_df = temp_df.dropna(subset=['Reviews'], axis=0,how='any',inplace=False)

    temp_df["Rating"] = temp_df["Rating"].astype(str)
    temp_df["Rating"] = temp_df["Rating"].astype(int)
    temp_df = temp_df.drop(temp_df[temp_df["Rating"]==3].index, inplace=False)
    # 3 - Neutral Rating or review, These are with rating of 3
    # This rating will be drop to be dataframe since these are all neither positive or negative
    return temp_df

# This will train/test all record in the database
# sqlstring_cm = "SELECT Reviews, Rating FROM gadget_reviews where Model='TestModel'"
temp_df_cm = pd.DataFrame(columns=["Reviews", "Rating", "Model", "Recommend"])
sqlstring_cm = "SELECT * FROM gadget_reviews"
temp_df_cm = pd.read_sql(sqlstring_cm, mysqlconn)

df = temp_df_cm[temp_df_cm['Reviews'].notna()]
df = sub_datacleaning(temp_df_cm)

df['Recommend'] = ''
df.loc[df['Rating'] >= 4, 'Recommend'] = "Recommend"
df.loc[df['Rating'] <= 2, 'Recommend'] = "Not Recommend"

label2id = {"Recommend": 1, "Not Recommend": 0}
id2label = {1: "Recommend", 0: "Not Recommend"}

# Tokenize and build vocabulary
def tokenize(sentence):
    return word_tokenize(sentence.lower())

vocab = set()
for reviews in df['Reviews']:
    vocab.update(tokenize(reviews))
for model1 in df['Model']:
    vocab.update(tokenize(model1))

word2idx = {word: i+1 for i, word in enumerate(vocab)}  # reserve 0 for padding
word2idx["<PAD>"] = 0

MAX_LEN = 30  # Max review length (truncate/pad to this)
def encode_text(text):
    tokens = tokenize(text)
    ids = [word2idx.get(t, 0) for t in tokens][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))

def encode_sentence(sentence):
    tokens = tokenize(sentence)
    ids = [word2idx.get(token, 0) for token in tokens][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))

# Dataset Class
class ReviewDataset(Dataset):
    def __init__(self, data):
        self.X = [encode_sentence(review) for review in df['Reviews']]
        self.ratings = [rating for rating in df['Rating']]
        self.y = [float(label2id[label]) for label in df['Recommend']]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float),
            torch.tensor(float(self.y[idx]), dtype=torch.float)
        )

# Model
class ProductLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim + 1, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, rating):
        embedded = self.embed(x)
        _, (h_n, _) = self.lstm(embedded)
        h_n = h_n.squeeze(0)
        combined = torch.cat((h_n, rating.unsqueeze(1)), dim=1)
        out = torch.relu(self.fc1(combined))
        return torch.sigmoid(self.fc2(out))

# Train/Test Split
train_data, test_data = train_test_split(df, test_size=0.25, random_state=42)
train_ds = ReviewDataset(train_data)
test_ds = ReviewDataset(test_data)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProductLSTM(len(word2idx), 64, 128).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for x, rating, y in train_loader:
        x, rating, y = x.to(device), rating.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x, rating).squeeze()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), 'LSTM_Model.pt')
print ("LSTM Model were created")
