import pandas as pd
import numpy as np

import torch
import torchtext
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
import re
import random
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import mysql.connector
from sqlalchemy import create_engine
import sqlalchemy as sqlalch

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
    
    # # Checking for missing values. Fill necessary and drop if reviews are null
    # if temp_df["Username"].isnull().values.any() == True:
    #     temp_df["Username"] = temp_df["Username"].fillna("No Username")       
    
    # # Date with invalid values will be default to 1/1/11, which also not useful :)
    # # Date with no values will also be converted, which also not useful :)
    # if temp_df["Date"].isnull().values.any() == True:
    #     temp_df["Date"] = temp_df["Date"].fillna("1/1/11")

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
    # temp_df["Rating"] = temp_df["Rating"].str.replace('[1-2]', '0', regex=True)
    # temp_df["Rating"] = temp_df["Rating"].str.replace('[4-5]', '1', regex=True)
    temp_df["Rating"] = temp_df["Rating"].astype(int)
    # 3 - Neutral Rating or review, These are with rating of 3
    # This rating will be drop to be dataframe since these are all neither positive or negative
    temp_df = temp_df.drop(temp_df[temp_df["Rating"]==3].index, inplace=False)
    return temp_df

# This will train/test all record in the database
# sqlstring_cm = "SELECT Reviews, Rating FROM gadget_reviews where Model='TestModel'"
sqlstring_cm = "SELECT Model, Reviews, Rating FROM gadget_reviews"
temp_df_cm = pd.read_sql(sqlstring_cm, mysqlconn)

df = temp_df_cm[temp_df_cm['Reviews'].notna()]
df = sub_datacleaning(temp_df_cm)
df['Recommend'] = ''
df.loc[df['Rating'] >= 4, 'Recommend'] = "Recommend"
df.loc[df['Rating'] <= 2, 'Recommend'] = "Not Recommend"

df['Rating'] = df['Rating'].apply(lambda x: round(x))
df['Reviews'] = df['Reviews'].apply(lambda x: x.lower())
df['Model'] = df['Model'].apply(lambda x: x.lower())

label2id = {"Recommend": 1, "Not Recommend": 0}
id2label = {1: "Recommend", 0: "Not Recommend"}

df['Rating'].unique()
df = df.sample(frac=1).reset_index(drop=True)

df = df[df['Reviews'].map(len) > 10]
print(len(df))
df = df.reset_index(drop=True)

df['Reviews'].map(len).max()

# training_df, testing_df = df.loc[:0.5*len(df)], df.loc[0.75*len(df):]
training_df, testing_df = train_test_split(df, test_size=.20, shuffle=True, random_state=42)
training_df.tail()
testing_df.tail()

# del data_df
# del df

training_df.to_csv("training.csv", index=False)
testing_df.to_csv("testing.csv", index=False)

tokenizer = lambda x: x.split()

TEXT = torchtext.data.Field(
    sequential=True, 
    tokenize=tokenizer, 
    lower=True, 
    include_lengths=True, 
    batch_first=True, 
    fix_length=200)
RATING = torchtext.data.LabelField(dtype=torch.float)
LABEL = torchtext.data.Field(
    sequential=True, 
    lower=True, 
    include_lengths=True, 
    batch_first=True, 
    fix_length=200)
MODEL = torchtext.data.Field(
    sequential=True, 
    lower=True, 
    include_lengths=True, 
    batch_first=True, 
    fix_length=200)

fields = [('Reviews',TEXT),('Rating', RATING),('Recommend', LABEL),('Model',MODEL)]

train_data = torchtext.data.TabularDataset("training.csv","csv", fields, skip_header=True)
test_data = torchtext.data.TabularDataset("testing.csv","csv", fields, skip_header=True)

train_data.examples[0].Reviews, train_data.examples[0].Rating, train_data.examples[0].Model

# del training_df
# del testing_df

# Word Embeddings
from gensim.models import Word2Vec
from torchtext.vocab import Vectors
tokenized_reviews = [word_tokenize(review.lower()) for review in df['Reviews'].tolist()]
wordvector_model = Word2Vec(tokenized_reviews, vector_size=100, min_count=1)
wordvector_model.wv.save_word2vec_format("custom.vec")
vectors = Vectors(name="custom.vec")

TEXT.build_vocab(train_data, vectors=torchtext.vocab.Vectors("custom.vec", cache = '../output/working/vector_cache'))
LABEL.build_vocab(train_data, vectors=torchtext.vocab.Vectors("custom.vec", cache = '../output/working/vector_cache'))
MODEL.build_vocab(train_data, vectors=torchtext.vocab.Vectors("custom.vec", cache = '../output/working/vector_cache'))
RATING.build_vocab(train_data)

# word_embeddings = LABEL.vocab.vectors
# word_embeddings = MODEL.vocab.vectors
word_embeddings = TEXT.vocab.vectors
word_embeddings.shape
train_data, valid_data = train_data.split()

train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train_data, valid_data, 
     test_data),
     batch_size=32,
     sort_key=lambda x: len(x.Reviews),
     repeat=False,
     shuffle=True)

vocab_size = len(TEXT.vocab)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(vocab_size, device)

word_embeddings.shape

torch.save(word_embeddings, "word_embeddings.pt")

import dill

with open("TEXT.Field", "wb") as f:
    dill.dump(TEXT, f)

class ClassifierModel(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(ClassifierModel, self).__init__()
        """
        output_size : 2 = (pos, neg)
        """
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)  # Initiale the look-up table.
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assign pre-trained GloVe word embedding.
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size=None):
        """ 
        final_output.shape = (batch_size, output_size)
        """
        input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cpu()) # Initial hidden state of the LSTM
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cpu()) # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cpu())
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cpu())
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_output = self.label(final_hidden_state[-1]) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)

        return final_output

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.to(device)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.Reviews[0]
        target = batch.Rating
        target = torch.autograd.Variable(target).long()

        if (text.size()[0] != 32):# One of the batch has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.Reviews[0]
            if (text.size()[0] != 32):
                continue
            target = batch.Rating
            target = torch.autograd.Variable(target).long()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)

batch_size = 32
output_size = 11
hidden_size = 256
embedding_length = 100
model = ClassifierModel(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

#architecture
print(model)

#No. of trianable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
print(f'The model has {count_parameters(model):,} trainable parameters')

learning_rate = 0.001
loss_fn = F.cross_entropy

for epoch in range(20):
    train_loss, train_acc = train_model(model, train_iter, epoch)
    val_loss, val_acc = eval_model(model, valid_iter)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

test_loss, test_acc = eval_model(model, test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

torch.save(model.state_dict(), 'saved_weights_model.pt')

def tokenize(sentence):
    return word_tokenize(sentence.lower())

def predict_for_product(product_name):
    product_name= "Galaxy Buds 3 Pro AI"

    # Collect all reviews for this product
    product_reviews = [(modelname, review, rating, label) for (modelname, review, rating, label) in df if p.lower() == product_name.lower()]
    if not product_reviews:
        return "Product not found."

    model.eval()
    predictions = []
    with torch.no_grad():
        for p, review, rating in product_reviews:
            x = torch.tensor([p + " " + review], dtype=torch.long).to(device)
            r = torch.tensor([rating], dtype=torch.float).to(device)
            out = model(x, r).item()
            predictions.append(out)

    avg_score = sum(predictions) / len(predictions)
    return "Recommend" if avg_score >= 0.5 else "Not Recommend"

# ENTER PRODUCT MODEL HERE
test_sent = "Galaxy Buds 3 Pro AI"
test_sent = "recommend"
result = predict_for_product(test_sent)
print(f"\nFinal Recommendation for '{test_sent}': {result}")


test_sent = MODEL.preprocess(test_sent)
test_sent = [[MODEL.vocab.stoi[x] for x in test_sent]]
test_sent = np.asarray(test_sent)
test_sent = torch.LongTensor(test_sent)
test_tensor = Variable(test_sent)
test_tensor = test_tensor.cpu()
model.eval()
output = model(test_tensor, 1)
out = F.softmax(output, 1)
print("Rating",torch.argmax(output[0]).item())

