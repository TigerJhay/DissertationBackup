import openai
API_KEY = ("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")

openai.api_key = API_KEY

#chat_log = []

while True:
    user_msg = input()
    if user_msg.lower() == "quit":
        break
    else:
        #chat_log.append({"role":"user", "content":user_msg})
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role":"user", "content":user_msg}]
        )
        message_response = response.choices[0].message.content.strip()
        print("The result is:", message_response.strip("\n").strip())


        # REFERENCE FOR LSTM FROM CHATGPT

        import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence

# Sample text and labels
texts = [
    "I love machine learning",
    "Deep learning is amazing",
    "LSTM networks are powerful",
    "Natural language processing is fun",
    "I enjoy learning about AI"
]
labels = [1, 1, 1, 0, 0]

# Tokenize
tokenized_texts = [sentence.lower().split() for sentence in texts]

# Train Word2Vec
w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=50, window=2, min_count=1, workers=4)

# Word2index dictionary
word2idx = {word: i+1 for i, word in enumerate(w2v_model.wv.index_to_key)}
word2idx["<PAD>"] = 0

# Embedding matrix
embedding_matrix = torch.zeros((len(word2idx), 50))
for word, idx in word2idx.items():
    if word != "<PAD>":
        embedding_matrix[idx] = torch.tensor(w2v_model.wv[word])

# Convert sentences to index sequences
indexed_texts = [[word2idx[word] for word in sent] for sent in tokenized_texts]
padded_texts = pad_sequence([torch.tensor(seq) for seq in indexed_texts], batch_first=True)
labels_tensor = torch.tensor(labels, dtype=torch.float32)

# Custom Dataset
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TextDataset(padded_texts, labels_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# LSTM Model using pretrained embeddings
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim):
        super(LSTMClassifier, self).__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return self.sigmoid(out).squeeze()

# Initialize model
model = LSTMClassifier(embedding_matrix, hidden_dim=64)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for X_batch, y_batch in dataloader:
        output = model(X_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
