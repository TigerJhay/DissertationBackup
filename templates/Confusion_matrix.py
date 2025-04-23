import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix_pytorch(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix for PyTorch predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def evaluate_lstm_model_pytorch(model, test_loader, label_encoder=None, device='cpu'):
    """
    Evaluates a PyTorch LSTM model and generates the confusion matrix and classification report.

    Args:
        model (nn.Module): Trained PyTorch LSTM model.
        test_loader (DataLoader): DataLoader for the test dataset.
        label_encoder (LabelEncoder, optional): Label encoder if labels are not numerical. Defaults to None.
        device (str, optional): Device to perform evaluation on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        None: Prints the confusion matrix and classification report.
    """
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if label_encoder:
        true_labels = label_encoder.inverse_transform(all_labels)
        predicted_labels = label_encoder.inverse_transform(all_preds)
        classes = label_encoder.classes_
    else:
        true_labels = np.array(all_labels)
        predicted_labels = np.array(all_preds)
        classes = np.unique(np.concatenate((true_labels, predicted_labels)))

    print("Confusion Matrix:")
    plot_confusion_matrix_pytorch(true_labels, predicted_labels, classes=classes)

    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=classes))

# Example Usage (assuming you have a trained PyTorch LSTM model):
if __name__ == '__main__':
    # Define device (use 'cuda' if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dummy data for demonstration
    texts = ["good movie", "bad movie", "excellent film", "terrible flick", "amazing acting", "awful performance"]
    labels = ["positive", "negative", "positive", "negative", "positive", "negative"]

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    # Tokenize text (simplified for example)
    tokenizer = torch.utils.data.Dataset  # Using a dummy tokenizer for simplicity
    vocab = sorted(list(set(" ".join(texts))))
    word_to_index = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(vocab)

    def text_to_indices(text):
        return [word_to_index[char] for char in text if char in word_to_index]

    sequences = [text_to_indices(text) for text in texts]
    # Pad sequences to the same length (simplified)
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = [seq + [0] * (max_len - len(seq)) for seq in sequences]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.3, random_state=42)

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2)

    # Define a simple LSTM model (PyTorch)
    class SimpleLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_units, output_size):
            super(SimpleLSTM, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_units, batch_first=True)
            self.fc = nn.Linear(hidden_units, output_size)

        def forward(self, x):
            embedded = self.embedding(x)
            out, _ = self.lstm(embedded)
            out = self.fc(out[:, -1, :]) # Take the output of the last time step
            return out

    # Instantiate the model
    embedding_dim = 5
    hidden_units = 10
    pytorch_model = SimpleLSTM(vocab_size, embedding_dim, hidden_units, num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.01)

    # Train the PyTorch model (simplified for demonstration)
    num_epochs = 2
    for epoch in range(num_epochs):
        pytorch_model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = pytorch_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate and get confusion matrix
    evaluate_lstm_model_pytorch(pytorch_model, test_loader, label_encoder=label_encoder, device=device)