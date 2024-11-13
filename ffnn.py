import json
import os
import random
import time
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

# fix random seeds
random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        # The rectified linear unit; one valid choice of activation function
        self.activation = nn.ReLU() 
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        # The softmax function that converts vectors into probability
        # distributions; computes log probabilities for computational benefits
        self.softmax = nn.LogSoftmax() 
        # The cross-entropy/negative log likelihood loss taught in class
        self.loss = nn.NLLLoss() 

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # obtain first hidden layer representation
        hidden = self.activation(self.W1(input_vector))
        # obtain output layer representation
        output = self.W2(hidden)
        # obtain probability dist.
        predicted_vector = self.softmax(output)

        return predicted_vector


# Returns:
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index), device=DEVICE)
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        test = json.load(test_f)

    tra = []
    val = []
    te = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in test:
        te.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val, te


def plot_loss(losses, filename='training_loss.png', dpi=300):
    plt.figure()
    plt.plot(range(len(losses)), losses, marker='o')
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_acc(accs, filename='valid_acc.png', dpi=300):
    plt.figure()
    plt.plot(range(len(accs)), accs, marker='o')
    plt.title('Validation Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()


def main(args, data, vocab):

    train_data = data["train_data"]
    valid_data = data["valid_data"]
    test_data = data["test_data"]

    epochs = args.epochs
    lr = args.lr
    hidden_dim = args.hidden_dim
    early_stop_patience = 5

    if args.sweep:
        wandb.init()
        hidden_dim = wandb.config.hidden_dim
        lr = wandb.config.lr

    model = FFNN(input_dim = len(vocab), h = hidden_dim).to(DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_loss_at_best_val = 0
    train_acc_at_best_val = 0
    best_train_acc = 0
    best_valid_loss = float('inf')
    best_valid_acc = 0

    train_losses = []
    valid_accuracies = []

    print(f"========== Training for {epochs} epochs ==========")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = None
        loss_total = 0
        loss_count = 0
        correct = 0
        total = 0
        start_time = time.time()
        print(f"Training started for epoch {epoch}")
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16 
        N = len(train_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            train_loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label], device=DEVICE))
                if train_loss is None:
                    train_loss = example_loss
                else:
                    train_loss += example_loss
            train_loss = train_loss / minibatch_size
            loss_total += train_loss.data
            loss_count += 1
            train_loss.backward()
            optimizer.step()

        train_loss = loss_total/loss_count
        train_losses.append(train_loss.detach().cpu().numpy())

        train_accuracy = correct / total
        # print(f"Training completed for epoch {epoch}")
        # print(f"Training accuracy for this epoch: {train_accuracy}")
        # print(f"Training time for this epoch: {time.time() - start_time}")


        with torch.no_grad():
            model.eval()
            valid_loss = None
            correct = 0
            total = 0
            start_time = time.time()
            # print("Validation started for epoch {}".format(epoch))
            minibatch_size = 16
            N = len(valid_data)
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                valid_loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label], device=DEVICE))
                    if valid_loss is None:
                        valid_loss = example_loss
                    else:
                        valid_loss += example_loss
                valid_loss = valid_loss / minibatch_size

        valid_accuracy = correct / total
        valid_accuracies.append(valid_accuracy)
        # print(f"Validation completed for epoch {epoch}")
        # print(f"Validation accuracy for this epoch {valid_accuracy}")
        # print(f"Validation time for this epoch: {time.time() - start_time}")

        print(f"Training accuracy for this epoch: {train_accuracy}")
        print(f"Validation accuracy for this epoch {valid_accuracy}")

        # the early stop code below wil stop the training if validation accuracy does not 
        # go up for for a few epochs specified by the patience
        if valid_accuracy < best_valid_acc and train_accuracy > best_train_acc:
            early_stop_patience -= 1
            print(f"early_stop_patience: {early_stop_patience}")
            if early_stop_patience == 0:
                print("Training stopped to avoid overfitting!")
                print(f"Best validation accuracy is: {best_valid_acc}")
                break

        best_train_acc = max(best_train_acc, train_accuracy)

        if best_valid_acc < valid_accuracy:
            train_loss_at_best_val = train_loss
            train_acc_at_best_val = train_accuracy
            best_valid_loss = valid_loss
            best_valid_acc = valid_accuracy
            torch.save(model.state_dict(), f'./models/ffnn/best_model_{hidden_dim}_{lr}.pth')


    # write out to results/test.out
    print("========== Testing ==========")
    # Load best model
    model.load_state_dict(torch.load(f'./models/ffnn/best_model_{hidden_dim}_{lr}.pth', weights_only=True))

    correct = 0
    total = 0
    start_time = time.time()
    minibatch_size = 16
    N = len(test_data)
    with torch.inference_mode():
        model.eval()  # Set model to evaluation mode
        for minibatch_index in tqdm(range(N // minibatch_size)):
            for example_index in range(minibatch_size):
                input_vector, gold_label = test_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1

    test_acc = correct/total
    print(f"Test accuracy: {test_acc}")


    plot_loss(train_losses)
    plot_acc(valid_accuracies)


    if args.sweep:
        wandb.log(
            {
                "train_loss": train_loss_at_best_val,
                "train_acc": train_acc_at_best_val,
                "val_acc": best_valid_acc,
                "val_loss": best_valid_loss,
                "test_acc": test_acc
            }
        )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, default = 64, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, default = 20, help = "num of epochs to train")
    parser.add_argument("--lr", type=float, default = 0.01, help = "learning rate")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", required = True, help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    # load data
    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    data = {
        "train_data": convert_to_vector_representation(train_data, word2index),
        "valid_data": convert_to_vector_representation(valid_data, word2index),
        "test_data": convert_to_vector_representation(test_data, word2index)
    }

    os.makedirs("./models/ffnn/", exist_ok=True)

    if args.sweep:
        sweep_configuration = {
            "method": "grid",
            "metric": {"goal": "maximize", "name": "val_acc"},
            "parameters": {
                "hidden_dim": {"values": [16, 32, 64, 128]},
                "lr": {"values": [1e-4, 1e-3, 1e-2]},
            },
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="ffnn")
        wandb.agent(sweep_id, function=lambda: main(args, data, vocab))

    else:
        main(args, data, vocab)
