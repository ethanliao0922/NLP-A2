# starter code for a2

Example for runnning the code:

**FFNN**

``python ffnn.py --train_data ./training.json --val_data ./validation.json --test_data ./test.json``

**RNN**

``python rnn.py --train_data ./training.json --val_data ./validation.json --test_data ./test.json``

Full set of arguments:

``"-hd"`` or ``"--hidden_dim"``, default = 64 "hidden_dim"

``"-e"`` or ``"--epochs"`` default = 20, "num of epochs to train"

``"--lr"``, default = 0.01, "learning rate"

``"--train_data"``, "path to training data"

``"--val_data"``, "path to validation data"

``"--test_data"``, "path to test data"

``"--sweep"``, "perform hyperparameter sweep. needs wandb account."
