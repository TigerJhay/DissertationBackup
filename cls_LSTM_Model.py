import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import mysql.connector
from sqlalchemy import create_engine
import sqlalchemy as sqlalch
import torch.nn as nn

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

# This will train/test all record in the database
sqlstring_cm = "SELECT * FROM gadget_reviews where brand='Samsung'"
temp_df_cm = pd.read_sql(sqlstring_cm, mysqlconn)
temp_df_cm = sub_datacleaning(temp_df_cm)

embedding_size = 50
SEQUENCE_LENGTH = 50
batch_size = 50
epochs = 5

temp_df_cm["Reviews"] = temp_df_cm["Reviews"].apply(word_tokenize)
df_train, df_test = train_test_split(temp_df_cm, test_size=.2, random_state=42)
all_reviews = df_train['Reviews'].tolist()
all_reviews.extend(df_test['Reviews'].tolist())

wordvector_model = Word2Vec(all_reviews, vector_size = embedding_size)
word2idx = {word: i+1 for i, word in enumerate(wordvector_model.wv.index_to_key)}
wordvector_model.wv[word2idx]

def convert_sequences_to_tensor(sequences, num_tokens_in_sequence, embedding_size):
    num_sequences = len(sequences)
    print((num_sequences, num_tokens_in_sequence, embedding_size))
    
    data_tensor = torch.zeros((num_sequences, num_tokens_in_sequence, embedding_size))
    
    for index, review in enumerate(list(sequences)):
        # Create a word embedding for each word in the review (where a review is a sequence)
        truncated_clean_review = review[:num_tokens_in_sequence] # truncate to sequence length limit
        list_of_word_embeddings = [wordvector_model.wv[word] if word in wordvector_model.wv else [0.0]*embedding_size for word in truncated_clean_review]

        # convert the review to a tensor
        sequence_tensor = torch.FloatTensor(list_of_word_embeddings)

        # add the review to our tensor of data
        review_length = sequence_tensor.shape[0] # (review_length, embedding_size)
        data_tensor[index,:review_length,:] = sequence_tensor
    
    return data_tensor

train_data_X = convert_sequences_to_tensor(df_train['Reviews'].to_numpy(), SEQUENCE_LENGTH, embedding_size)
train_data_y = torch.FloatTensor([int(d) for d in df_train['Rating'].to_numpy()])
test_data_X = convert_sequences_to_tensor(df_test['Reviews'].to_numpy(), SEQUENCE_LENGTH, embedding_size)
test_data_y = torch.FloatTensor([int(d) for d in df_test['Rating'].to_numpy()])

train_data = TensorDataset(train_data_X, train_data_y)
test_data = TensorDataset(test_data_X, test_data_y)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

#Need to use the resources of CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device
    

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, drop_prob=0.7):
        super(LSTMModel,self).__init__()

        self.num_stacked_layers = num_stacked_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_stacked_layers,
            batch_first = True)
        
        self.dropout = nn.Dropout(drop_prob) # randomly sets outputs of a tensor to 0 during training
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize the cell state and hidden state
        h0 = torch.zeros((self.num_stacked_layers, batch_size, self.hidden_size)).to(device)
        c0 = torch.zeros((self.num_stacked_layers, batch_size, self.hidden_size)).to(device)

        # Call the LSTM
        lstm_out, hidden = self.lstm(x, (h0, c0))

        # contiguous() moves all data into 1 block of memory on the GPU
        # (batch_size, sequence_size, embedding_size) -> (batch_size*sequence_size, embedding_size)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)

        # dropout and fully connected layer
        lstm_out = self.dropout(lstm_out) # Only during training
        fc_out = self.fc(lstm_out)
        
        # apply the sigmoid function to maps the value to somewhere between 0 and 1
        sigmoid_out = self.sigmoid(fc_out)

        # reshape to be batch_size first - every batch has a value between 0 and 1
        sigmoid_out = sigmoid_out.view(batch_size, -1) # a list of lists with single elements
        sigmoid_out = sigmoid_out[:, -1] # get the output labels as a list

        # return last sigmoid output and hidden state
        return sigmoid_out, hidden 

LSTM_INPUT_SIZE = embedding_size # size of the embeddings
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_STACKED_LAYERS = 2

lstm_model = LSTMModel(LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, LSTM_NUM_STACKED_LAYERS)
lstm_model.to(device)
# print(lstm_model)

lr=0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)

def accuracy(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

def train_loop(model, train_loader, optimizer, criterion):
    model.train()
    train_accuracy = 0.0
    train_losses = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs, h = model(inputs) # Forward pass
        loss = criterion(outputs, labels) # Calculate the loss
        optimizer.zero_grad() # Clear out all previous gradients
        loss.backward() # Calculate new gradients
        optimizer.step() # Update parametres using the gradients

        train_losses.append(loss.item())
        train_accuracy += accuracy(outputs, labels)

        epoch_train_loss = np.mean(train_losses) * 100.00
        epoch_train_acc = (train_accuracy/len(train_loader.dataset))*100.0
    return (epoch_train_loss, epoch_train_acc)

# Test/Validation Loop
def test_loop(model, test_loader, criterion):
    model.eval()
    test_accuracy = 0.0
    test_losses = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, val_h = model(inputs)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
            test_accuracy += accuracy(outputs, labels)

    epoch_test_loss = np.mean(test_losses) * 100.0
    epoch_test_accuracy = (test_accuracy/len(test_loader.dataset))*100.0

    return (epoch_test_loss, epoch_test_accuracy)

# Training and validation loop
epoch_train_losses = []
epoch_train_accs = []
epoch_test_losses = []
epoch_test_accs = []

for epoch in range(epochs):
    epoch_train_loss, epoch_train_acc = train_loop(lstm_model, train_loader, optimizer, criterion)
    train_loss = round(epoch_train_loss,3)
    train_accs = round(epoch_train_acc,3)

    epoch_test_loss, epoch_test_acc = test_loop(lstm_model, test_loader, criterion)
    test_loss = round(epoch_test_loss,3)
    test_accs = round(epoch_test_acc,3)

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f} Train Acc: {epoch_train_acc:.4f} | Test Loss: {epoch_test_loss:.4f} Test Acc: {epoch_test_acc:.4f}')
    epoch_train_losses.append(epoch_train_loss)
    epoch_train_accs.append(epoch_train_acc)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_accs.append(epoch_test_acc)

# Model Integration

# # Pytorch model
torch.save(lstm_model.state_dict(), 'actual_LSTM_model.pth')
print("Model saved!")


loaded_model =  LSTMModel(50,128,2)
loaded_model.load_state_dict(torch.load('actual_LSTM_model.pth'))
loaded_model.eval()

def recommended_next(input_sequence):
    input_tensor = torch.tensor(input_sequence).unsqueeze(0)  # shape: [1, seq_len]
    with torch.no_grad():
        output = loaded_model(input_tensor)
        predicted_item = torch.argmax(output, dim=1).item()
    return predicted_item

test_val = [1,2]
next_item = recommended_next(test_val)
print("Recommended next item ID:", next_item)
# model_new = torch.load('actual_LSTM_model.pth')
# y_pred = model_new("Test")
# lstm_model = LSTMModel()
# lstm_model.load_state_dict(torch.load('model.pth')
