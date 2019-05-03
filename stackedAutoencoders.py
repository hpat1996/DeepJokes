import sys
import math
import numpy as np

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.parallel

import torch.optim as optim
import torch.utils.data

# Use GPU if available
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants for the data set 
DATA_FILE           = "data/jester-data.csv"

UNKNOWN_RATING      = 99
MIN_RATING          = -10
MAX_RATING          = 10
DESIRED_NUM_RATING  = 5

NUM_TRAIN           = (0.7, 0.7)

# Hyperparameters for the model
LEARNING_RATE       = 0.01
WEIGHT_DECAY        = 0.0

NUM_ITERATIONS      = 200

print("\n")
print("Initializing...")
# Normalize the ratings in the data set.
# Unknown rating -> 0
# Known ratings -> [1, DESIRED_NUM_RATING]
def normalizeData(n):
    if n == UNKNOWN_RATING:
        return 0
    mid = (MAX_RATING - MIN_RATING) / 2
    return math.ceil((n + mid) / (DESIRED_NUM_RATING - 1))

# Load the data from the file
# Discard first column as it is not useful
data = np.loadtxt(DATA_FILE, dtype=np.float, delimiter=",")[:, 1:]

# Normalize the data
data = np.vectorize(normalizeData)(data)
num_users, num_jokes = data.shape

# Divide the data into train and test
num_train_users   = int(NUM_TRAIN[0] * num_users)
num_train_jokes   = int(NUM_TRAIN[1] * num_jokes)

train_data  = np.zeros(shape=data.shape)
test_data   = np.zeros(shape=data.shape)

train_data  [               :               ,                       :num_train_jokes]   = data[               :               ,                       :num_train_jokes] 
train_data  [               :num_train_users,                       :               ]   = data[               :num_train_users,                       :               ] 
test_data   [num_train_users:               ,   num_train_jokes     :               ]   = data[num_train_users:               ,   num_train_jokes     :               ]


train_data  = torch.tensor(train_data, device = DEVICE, dtype=torch.float)
test_data   = torch.tensor(test_data, device = DEVICE, dtype=torch.float)

# The stacked auto encoder model
class StackedAutoEncoder(nn.Module):
    def __init__(self, input_dim = num_jokes, hidden_dim = 5, output_dim = num_jokes, num_stack = 4):
        super(StackedAutoEncoder, self).__init__()

        self.ae1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.ae2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.ae3 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.ae4 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.ae5 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.ae6 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.ae1(x)
        x = self.ae2(x)
        x = self.ae3(x)
        x = self.ae4(x)
        x = self.ae5(x)
        x = self.ae6(x)
        return x

# MSE Loss function
def MSEloss(predicted, actual):
    # Get the mask
    mask = actual != 0
    mask = mask.float()

    # Mask the columns in the output where the input is unrated
    predicted = predicted * mask

    # Total number of ratings
    num_ratings = torch.sum(mask)

    # Calculate the square of the errors
    error = torch.sum((actual - predicted) ** 2)
    return error, num_ratings




mode = sys.argv[1]

if (mode == 'train'):
    # Training on train data
    stackedAutoEncoder  = StackedAutoEncoder().to(DEVICE)
    optimizer           = optim.Adam(stackedAutoEncoder.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

    print("Training...")
    # Train the model
    for i in range(NUM_ITERATIONS):
        predicted_ratings = stackedAutoEncoder(train_data)

        optimizer.zero_grad() 
        loss, num_ratings = MSEloss(predicted_ratings, train_data)
        loss = loss / num_ratings
        loss.backward()
        optimizer.step()

        print("Epoch #", (i + 1), ": Training loss: ", loss.data.item())

    print("Training finished.\n")

    print("Testing...")
    predicted_ratings = stackedAutoEncoder(test_data)
    loss, num_ratings = MSEloss(predicted_ratings, test_data)
    loss = loss / num_ratings
    print("Loss on test data: ", loss.data.item())
    print("\n")

    print("Saving model...")
    torch.save(stackedAutoEncoder, "model")
    print("Saved model.")
    

elif (mode == 'test'):
    # Testing on test data
    print("Loading model...")
    stackedAutoEncoder = torch.load("model")
    print("Loaded model.")
    print("Testing...")

    predicted_ratings = stackedAutoEncoder(test_data)
    loss, num_ratings = MSEloss(predicted_ratings, test_data)
    loss = loss / num_ratings
    
    print("Loss on test data: ", loss.data.item())

else:
    print("Usage: python3 stackedAutoencoders.py <train | test>")

print('\n')