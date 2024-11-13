import json
import os
import pickle
import random
import string
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
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        _, hidden = self.rnn(inputs)
        # obtain output layer representations
        output = self.W(hidden[-1])
        # obtain probability dist.
        predicted_vector = self.softmax(output)

        return predicted_vector


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


def main(args, data):

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this
    train_data = data["train_data"]
    valid_data = data["valid_data"]
    test_data = data["test_data"]
    word_embedding = pickle.load(open('./Data_Embedding/word_embedding.pkl', 'rb'))

    epochs = args.epochs
    lr = args.lr
    hidden_dim = args.hidden_dim

    if args.sweep:
        wandb.init()
        hidden_dim = wandb.config.hidden_dim
        lr = wandb.config.lr

    model = RNN(50, hidden_dim).to(DEVICE)  # Fill in parameters
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    stopping_condition = False
    epoch = 0

    last_train_acc = 0
    last_valid_acc = 0

    train_losses = []
    valid_accuracies = []

    for epoch in range(epochs):
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        print(f"Training started for epoch {epoch}")
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            train_loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]

                # Transform the input into required shape
                vectors = torch.tensor(vectors, device=DEVICE).view(len(vectors), 1, -1)
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label], device=DEVICE))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                total += 1
                if train_loss is None:
                    train_loss = example_loss
                else:
                    train_loss += example_loss

            train_loss = train_loss / minibatch_size
            loss_total += train_loss.data
            loss_count += 1
            train_loss.backward()
            optimizer.step()

        train_accuracy = correct / total
        train_loss = loss_total/loss_count
        train_losses.append(train_loss.detach().cpu().numpy())
        # print(f"Training completed for epoch {epoch}")
        # print(f"Training accuracy for this epoch: {train_accuracy}")

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            # print(f"Validation started for epoch {epoch}")
            for input_words, gold_label in tqdm(valid_data):
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                        in input_words]

                vectors = torch.tensor(vectors, device=DEVICE).view(len(vectors), 1, -1)
                output = model(vectors)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
                # print(predicted_label, gold_label)

        valid_accuracy = correct/total
        valid_accuracies.append(valid_accuracy)
        # print(f"Validation completed for epoch {epoch}")
        print(f"Training accuracy for this epoch: {train_accuracy}")
        print(f"Validation accuracy for this epoch: {valid_accuracy}")
            

        if valid_accuracy < last_valid_acc and train_accuracy > last_train_acc:
            # print("Training stopped to avoid overfitting!")
            # print(f"Best validation accuracy is: {last_valid_acc}")
            # break
            pass
        else:
            last_valid_acc = valid_accuracy
            last_train_acc = train_accuracy
            torch.save(model.state_dict(), f'./models/rnn/best_model_{hidden_dim}_{lr}.pth')

    print("========== Testing ==========")
    model.load_state_dict(torch.load(f'./models/rnn/best_model_{hidden_dim}_{lr}.pth', weights_only=True))

    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for input_words, gold_label in tqdm(test_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                    in input_words]

            vectors = torch.tensor(vectors, device=DEVICE).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1

        test_acc = correct/total
        print(f"Test accuracy: {test_acc}")

        plot_loss(train_losses)
        plot_acc(valid_accuracies)


    if args.sweep:
        wandb.log(
            {
                "train_acc": last_train_acc,
                "val_acc": last_valid_acc,
                "test_acc": test_acc
            }
        )

    # You may find it beneficial to keep track of training accuracy or training loss;

    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance


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

    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    data = {
        "train_data": train_data,
        "valid_data": valid_data,
        "test_data": test_data
    }

    print(
        "training data size: ",len(train_data),
        "\nvalidation data size: ", len(valid_data),
        "\ntest data size: ", len(test_data)
    )

    os.makedirs("./models/rnn/", exist_ok=True)

    if args.sweep:
        sweep_configuration = {
            "method": "grid",
            "metric": {"goal": "maximize", "name": "val_acc"},
            "parameters": {
                "hidden_dim": {"values": [16, 32, 64, 128]},
                "lr": {"values": [1e-4, 1e-3, 1e-2]},
            },
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="rnn")
        wandb.agent(sweep_id, function=lambda: main(args, data))

    else:
        main(args, data)
