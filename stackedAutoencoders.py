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
DESIRED_NUM_RATING  = 4

NUM_TRAIN           = 0.7
NUM_DEV             = 0.1
NUM_TEST            = 0.2

# Hyperparameters for the model
LEARNING_RATE       = 0.001
WEIGHT_DECAY        = 0.8

NUM_ITERATIONS      = 100


# Normalize the ratings in the data set.
# Unknown rating -> 0
# Known ratings -> [1, 1 + DESIRED_NUM_RATING]
def normalizeData(n):
    if n == UNKNOWN_RATING:
        return 0
    mid = (MAX_RATING - MIN_RATING) / 2
    return math.ceil((n + mid) / DESIRED_NUM_RATING)

# Load the data from the file
# Discard first column as it is not useful
data = np.loadtxt(DATA_FILE, dtype=np.float, delimiter=",")[:, 1:]

# Normalize the data
data = np.vectorize(normalizeData)(data)
num_users, num_jokes = data.shape

# Divide the data into train, dev and test
num_train   = int(NUM_TRAIN * num_jokes)
num_dev     = int(NUM_DEV   * num_jokes)
num_test    = int(NUM_TEST  * num_jokes) 

train_data  = np.zeros(shape=data.shape)
dev_data    = np.zeros(shape=data.shape)
test_data   = np.zeros(shape=data.shape)

train_data  [:,                     :num_train]                     = data[:,                     :num_train]
dev_data    [:, num_train           :num_train+num_dev]             = data[:, num_train           :num_train+num_dev]
test_data   [:, num_train+num_dev   :num_train+num_dev+num_test]    = data[:, num_train+num_dev   :num_train+num_dev+num_test]


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
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.ae3 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.ae4 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.ae1(x)
        x = self.ae2(x)
        x = self.ae3(x)
        x = self.ae4(x)
        return x

def MSEloss(predicted, actual, size_average=False):
    mask = actual != 0
    num_ratings = torch.sum(mask.float())
    criterion = nn.MSELoss(reduction='sum' if not size_average else 'mean')
    return criterion(predicted * mask.float(), actual), Variable(torch.Tensor([1.0])) if size_average else num_ratings

stackedAutoEncoder  = StackedAutoEncoder().to(DEVICE)
train_data          = torch.tensor(train_data, device = DEVICE, dtype=torch.float)
optimizer           = optim.Adam(stackedAutoEncoder.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)

# Train the mdoel
for i in range(NUM_ITERATIONS):
    train_loss = 0.0
    for user in range(num_users):
        actual_ratings = Variable(train_data[user], requires_grad=False)
        predicted_ratings = stackedAutoEncoder(actual_ratings)

        optimizer.zero_grad()
        loss, num_ratings = MSEloss(predicted_ratings, actual_ratings)
        loss = loss / num_ratings
        train_loss += loss.data.item()

        loss.backward()
        optimizer.step()

    print('Epoch #', (i + 1), ': Training loss: ', train_loss)
