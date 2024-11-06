import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import GloVe
import json
import string
import random
from tqdm import tqdm
from argparse import ArgumentParser

# Load GloVe vectors
glove = GloVe(name='6B', dim=50)  # Load 50-dimensional GloVe vectors. Adjust dim as needed.

class RNN(nn.Module):
    def __init__(self, input_dim, h, vocab_size, embed_dim, pretrained_embeddings):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(pretrained_embeddings, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Freeze embeddings (set True if you want to fine-tune)
        self.rnn = nn.RNN(embed_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        embedded = self.embedding(inputs)  # Convert input tokens to embeddings
        rnn_output, hidden = self.rnn(embedded)
        output_layer = self.W(rnn_output)
        summed_output = output_layer.sum(dim=0)
        predicted_vector = self.softmax(summed_output)
        return predicted_vector

def build_vocab_and_embeddings(train_data):
    vocab = {}
    embedding_matrix = []

    for text, _ in train_data:
        for word in text:
            if word.lower() not in vocab:
                vocab[word.lower()] = len(vocab)

    # Create an embedding matrix for the words in the vocabulary
    for word, idx in vocab.items():
        if word in glove.stoi:  # Check if the word exists in GloVe's vocabulary
            embedding_matrix.append(glove[word].tolist())
        else:
            embedding_matrix.append([0.0] * glove.dim)  # Use zero vector for unknown words

    return vocab, embedding_matrix

def tokenize_and_convert_to_ids(text, vocab):
    words = text.translate(str.maketrans("", "", string.punctuation)).split()
    return [vocab[word.lower()] if word.lower() in vocab else vocab['<unk>'] for word in words]

def load_data(train_data_path, val_data_path):
    with open(train_data_path) as training_f:
        training = json.load(training_f)
    with open(val_data_path) as valid_f:
        validation = json.load(valid_f)

    tra = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in training]
    val = [(elt["text"].split(), int(elt["stars"] - 1)) for elt in validation]
    return tra, val

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)

    print("========== Building vocabulary and embedding matrix ==========")
    vocab, embedding_matrix = build_vocab_and_embeddings(train_data)
    vocab_size = len(vocab)
    embed_dim = len(embedding_matrix[0])

    print("========== Initializing Model ==========")
    model = RNN(embed_dim, args.hidden_dim, vocab_size, embed_dim, embedding_matrix).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Example training loop (minimal)
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(train_data)
        for input_words, gold_label in train_data:
            optimizer.zero_grad()
            token_ids = torch.tensor(tokenize_and_convert_to_ids(" ".join(input_words), vocab), dtype=torch.long).unsqueeze(1).to(model.embedding.weight.device)
            output = model(token_ids)
            loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label], device=model.embedding.weight.device))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed")
