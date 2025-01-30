import pandas as pd
import numpy as np
from numpy import array
import re
import nltk 

#from wordcloud import wordcloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.cluster import KMeans

#import keras
#import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.dates import MonthLocator, DateFormatter, YearLocator
from tensorflow.keras.models import Sequential

# stopwords1 = set(STOPWORDS)
# new_words = ['ref','referee']
# new_stopwords = stopwords.union(new_words)

#custom stopwords, words that are not on nltk.stopwords. These words are not essential in reviews of gadgets.
custom_stopwords = ['also', 'dad', 'mom', 'kids', 'christmas', 'hoping']

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

#Access and load the dataset record of reviews
#df_reviews = pd.read_csv("./templates/Amazon_Review.csv")
df_reviews = pd.read_csv("./templates/Main_Dataset.csv")
df_reviews.head(20)

df_reviews['Reviews'] = df_reviews['Reviews'].str.lower()

# Checking for missing values. Fill necessary and drop if reviews are null
if df_reviews["Username"].isnull().values.any() == True:
    df_reviews["Username"] = df_reviews["Username"].fillna("No Username")       
if df_reviews["Date"].isnull().values.any() == True:
    df_reviews["Date"] = df_reviews["Date"].fillna("1/1/11")
if df_reviews["Reviews"].isnull().values.any() == True:
    df_reviews = df_reviews.dropna(subset=['Reviews'], axis=0,how='any',inplace=False)

#Remove Column Username since this column is unnecessary
df_reviews.drop(['Username'],axis='columns',inplace=True)

#Replace special tags inside sentiment
df_reviews["Reviews"] = df_reviews["Reviews"].str.replace("\n",' ')
df_reviews["Reviews"] = df_reviews["Reviews"].str.replace("\r",' ')

#Removal of URL and Links inside of reviews column
df_reviews = df_reviews.replace(r'http\S+', '', regex=True)
df_reviews = df_reviews.replace(r"x000D", '', regex=True)

#HTML tag removal
df_reviews = df_reviews.replace(r'<[^>]+>', '', regex= True)

#Punctuation and character removal
df_reviews = df_reviews.replace('[^a-zA-Z0-9]', ' ', regex=True)

#Single Character Removal
df_reviews = df_reviews.replace(r"\s+[a-zA-Z]\s+", ' ', regex=True)

#Multiple Spaces Removal
df_reviews = df_reviews.replace(r" +", ' ', regex=True)

#Stopword Removal
df_reviews = df_reviews.replace(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*','', regex=True)
df_reviews = df_reviews.replace(r'\b(' + r'|'.join(custom_stopwords) + r')\b\s*','', regex=True)

#Lemmatize dataframe, do I still need it???
def lemmatize_review(review_text):
    words = nltk.word_tokenize(review_text)
    lemmatize_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatize_text = ' '.join(lemmatize_words)
    return lemmatize_text
df_reviews['Reviews'] = df_reviews['Reviews'].apply(lemmatize_review)

#Rating of the sentiments will be converted into 3 classes
# 0 - Negative Rating or review, These are with rating of 1 & 2
# 1 - Positive Rating or review, These are with rating of 4 & 5
# 3 - Neutral Rating or review, These are with rating of 3
df_reviews["Rating"] = df_reviews["Rating"].astype(str)
df_reviews["Rating"] = df_reviews["Rating"].str.replace('[1-2]', '0', regex=True)
df_reviews["Rating"] = df_reviews["Rating"].str.replace('[3-5]', '1', regex=True)

df_naivebayes = pd.DataFrame(df_reviews)
df_lstm = pd.DataFrame(df_reviews)
df_kmeans = pd.DataFrame(df_reviews)

#----------------------------------------------------------
#This portion is part of Naive Bayes, Multinomial Algorithm
#----------------------------------------------------------

#Vectorize process
#vectorize = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii')
vectorize = CountVectorizer()

y_val = df_naivebayes['Rating']
x_val = df_naivebayes['Reviews']
x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=0.2, random_state=0)
x_train_count = vectorize.fit_transform(x_train.values)
x_train_count.toarray()

classifier = naive_bayes.MultinomialNB()
classifier.fit(x_train_count, y_train)

#no.array() should be use with predicttion dataset, values encoded are just for testing of algorithm
gadget_review_array = np.array(["Phone doesnt work","Capacity are bad", "Features are good"])
gadget_review_vector = vectorize.transform(gadget_review_array)
classifier.predict(gadget_review_vector)


#---------------------------------------
# This portion is for LSTM algorithm
#---------------------------------------

#Tokenize all words in the dataframe
df_lstm["Reviews"] = df_reviews["Reviews"].apply(word_tokenize)

df_train, df_test = train_test_split(df_lstm, test_size=.2)
df_train['Rating'].value_counts()
df_test['Rating'].value_counts()


from gensim.models import Word2Vec

embedding_size = 50
all_reviews = df_train['Reviews'].tolist()
all_reviews.extend(df_test['Reviews'].tolist())
#all_reviews

wordvector_model = Word2Vec(all_reviews, vector_size=50)

#wv['_____'] the value inside wv is the value needed for prediction
wordvector_model.wv['phone']
wordvector_model.wv.most_similar('phone', topn=3)

import torch
from torch.utils.data import DataLoader, TensorDataset

# cap each review to 100 words (tokens)
SEQUENCE_LENGTH = 100

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

print("Example Sequence:")
print(train_data_X[0])
print("Example Label:")
print(train_data_y[0])

train_data = TensorDataset(train_data_X, train_data_y)
test_data = TensorDataset(test_data_X, test_data_y)

batch_size = 5
#int(input("Enter value for batch size: "))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

#Need to use the resources of CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
     
import torch.nn as nn
class LSTMModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_stacked_layers, drop_prob=0.7):
    super(LSTMModel,self).__init__()

    self.num_stacked_layers = num_stacked_layers
    self.hidden_size = hidden_size

    self.lstm = nn.LSTM(
        input_size = input_size,
        hidden_size = hidden_size,
        num_layers = num_stacked_layers,
        batch_first = True
      )

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
print(lstm_model)

lr=0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)
epochs = 10
#int(input("Enter value for epochs"))

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

  epoch_train_loss = np.mean(train_losses)
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

  epoch_test_loss = np.mean(test_losses)
  epoch_test_accuracy = (test_accuracy/len(test_loader.dataset))*100.0

  return (epoch_test_loss, epoch_test_accuracy)

# Training and validation loop
epoch_train_losses = []
epoch_train_accs = []
epoch_test_losses = []
epoch_test_accs = []
for epoch in range(epochs):
  epoch_train_loss, epoch_train_acc = train_loop(lstm_model, train_loader, optimizer, criterion)
  epoch_test_loss, epoch_test_acc = test_loop(lstm_model, test_loader, criterion)

  print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f} Train Acc: {epoch_train_acc:.4f} | Test Loss: {epoch_test_loss:.4f} Test Acc: {epoch_test_acc:.4f}')

  epoch_train_losses.append(epoch_train_loss)
  epoch_train_accs.append(epoch_train_acc)
  epoch_test_losses.append(epoch_test_loss)
  epoch_test_accs.append(epoch_test_acc)

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10, 3))

plt.subplot(1, 2, 1)
plt.plot(epoch_train_accs, label='Train Accuracy')
plt.plot(epoch_test_accs, label='Test Accuracy')
plt.title("Accuracy")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(epoch_train_losses, label='Train Loss')
plt.plot(epoch_test_losses, label='Test Loss')
plt.title("Loss")
plt.legend()
plt.grid()
plt.show()


#----------------------------------------------
# This portion is for Cluster K-Means Algorithm
#----------------------------------------------

df_kmeans["Reviews"] = df_kmeans["Reviews"].values.astype("U")
vectorize = TfidfVectorizer(stop_words='english')
vectorized_value = vectorize.fit_transform(df_kmeans["Reviews"])

k_value = 10
k_model = KMeans(n_clusters=k_value, init='k-means++', max_iter=100, n_init=1)
k_model.fit(vectorized_value)

df_kmeans["clusters"] = k_model.labels_
df_kmeans.head()


# cluster_groupby = df_kmeans.groupby("clusters")
# for cluster in cluster_groupby.groups:
#     f = open("cluster"+str(cluster)+".csv","w")
#     data = cluster_groupby.get_group(cluster)[["Rating", "Reviews"]]
#     f.write(data.to_csv(index_label="id"))
#     f.close()

center_gravity = k_model.cluster_centers_.argsort()[:,::-1]
terms = vectorize.get_feature_names_out()

for ctr in range(k_value):
    print ("Cluster %d: " % ctr)
    for ctr2 in center_gravity[ctr, :10]:
        print ("%s" % terms[ctr2])
    print ("---------------------")

plt.scatter(df_kmeans['Reviews'], df_kmeans['clusters'])
plt.xlabel('clusters')
plt.ylabel('Reviews')
plt.show()