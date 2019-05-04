import sys
import numpy as np

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.parallel

import torch.optim as optim
import torch.utils.data

####################################################################################################

# Use GPU if available
DEVICE                      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants for the data set 
DATASET_FILE                = "data/jester-data.csv"

DATASET_UNKNOWN_RATING      = 99
DATASET_MIN_RATING          = -10
DATASET_MAX_RATING          = 10

NORMALIZED_UNKNOWN_RATING   = 0
NORMALIZED_MIN_RATING       = 1
NORMALIZED_MAX_RATING       = 5
NORMALIZED_ROUNDED          = True

                            # (% users, % jokes)
NUM_TRAIN                   = (0.7, 0.7)

# Hyperparameters for the model
LEARNING_RATE               = 0.01
WEIGHT_DECAY                = 0.0
LOSS_FUNCTION               = 'RMSE'
NUM_ITERATIONS              = 200

print("\n")
print("Initializing...")

####################################################################################################

# Normalize the ratings in the data set.
def normalizeData(n):
    if n == DATASET_UNKNOWN_RATING:
        return NORMALIZED_UNKNOWN_RATING

    n = round(n)

    dataset_range       = (DATASET_MAX_RATING    -  DATASET_MIN_RATING)
    normalized_range    = (NORMALIZED_MAX_RATING -  NORMALIZED_MIN_RATING)
    
    normalized_n = (((n - DATASET_MIN_RATING) * normalized_range) / dataset_range) + NORMALIZED_MIN_RATING

    if NORMALIZED_ROUNDED:
        normalized_n = round(normalized_n)

    return normalized_n

# Load the data from the file
# Discard first column as it is not useful
data = np.loadtxt(DATASET_FILE, dtype=np.float, delimiter=",")[:, 1:]

# Normalize the data
data = np.vectorize(normalizeData)(data)
num_users, num_jokes = data.shape

print(data)

# Divide the data into train and test
num_train_users   = int(NUM_TRAIN[0] * num_users)
num_train_jokes   = int(NUM_TRAIN[1] * num_jokes)

train_data  = np.zeros(shape=data.shape)
test_data   = np.zeros(shape=data.shape)

train_data  [               :               ,                       :num_train_jokes]   = data[               :               ,                       :num_train_jokes] 
train_data  [               :num_train_users,                       :               ]   = data[               :num_train_users,                       :               ] 
test_data   [num_train_users:               ,   num_train_jokes     :               ]   = data[num_train_users:               ,   num_train_jokes     :               ]


train_data  = torch.tensor(train_data,  device = DEVICE, dtype=torch.float)
test_data   = torch.tensor(test_data,   device = DEVICE, dtype=torch.float)

####################################################################################################

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

####################################################################################################

# MSE Loss function
def MSE_Loss(predicted, actual):
    # Get the mask
    mask = actual != NORMALIZED_UNKNOWN_RATING
    mask = mask.float()

    # Mask the columns in the output where the input is unrated
    predicted = predicted * mask

    # Total number of ratings
    num_ratings = torch.sum(mask)

    # Calculate the square of the errors
    error = torch.sum((actual - predicted) ** 2)
    return error, num_ratings

# RMSE Loss function
def RMSE_Loss(predicted, actual):
    error, num_ratings = MSE_Loss(predicted, actual)
    return (error / num_ratings) ** 0.5

def getLoss(predicted, actual, loss_function='MSE'):
    if (loss_function == 'MSE'):
        error, num_ratings = MSE_Loss(predicted, actual)
        return error / num_ratings
    elif (loss_function == 'RMSE'):
        return RMSE_Loss(predicted, actual)

####################################################################################################

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
        loss = getLoss(predicted_ratings, train_data, LOSS_FUNCTION)
        loss.backward()
        optimizer.step()

        print("Epoch #", (i + 1), ": Training loss: ", loss.data.item())

    print("Training finished.\n")

    print("Testing...")
    predicted_ratings = stackedAutoEncoder(test_data)
    loss = getLoss(predicted_ratings, test_data, LOSS_FUNCTION)
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
    loss = getLoss(predicted_ratings, test_data, LOSS_FUNCTION)
    
    print("Loss on test data: ", loss.data.item())


else:
    print("Usage: python3 stackedAutoencoders.py <train | test>")

print('\n')
