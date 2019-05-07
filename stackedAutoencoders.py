import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.ioff()

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.parallel

import torch.optim as optim
import torch.utils.data

####################################################################################################
# CONSTANTS AND HYPERPARAMETERS
####################################################################################################

# Use GPU if available
DEVICE                      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants for the data set 
DATASET_FILE                = "data/jester-data.csv"

DATASET_UNKNOWN_RATING      = 99
DATASET_MIN_RATING          = -10
DATASET_MAX_RATING          = 10

NORMALIZED_UNKNOWN_RATING   = 99
NORMALIZED_MIN_RATING       = 1
NORMALIZED_MAX_RATING       = 5
NORMALIZED_ROUNDED          = True

                            # (% users, % jokes)
NUM_TRAIN                   = (0.7, 0.7)

# Hyperparameters for the model
LEARNING_RATE               = 0.01
WEIGHT_DECAY                = 0.0
LOSS_FUNCTION               = 'MSE'
NUM_ITERATIONS              = 100

print("\n")
print("Initializing...")

####################################################################################################
# DATASET
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


# Divide the data into train and test
num_train_users   = int(NUM_TRAIN[0] * num_users)
num_train_jokes   = int(NUM_TRAIN[1] * num_jokes)

train_data  = np.zeros(shape=data.shape)
train_data.fill(NORMALIZED_UNKNOWN_RATING)
test_data   = np.zeros(shape=data.shape)
test_data.fill(NORMALIZED_UNKNOWN_RATING)

train_data  [               :               ,                       :num_train_jokes]   = data[               :               ,                       :num_train_jokes] 
train_data  [               :num_train_users,                       :               ]   = data[               :num_train_users,                       :               ] 
test_data   [num_train_users:               ,   num_train_jokes     :               ]   = data[num_train_users:               ,   num_train_jokes     :               ]


train_data  = torch.tensor(train_data,  device = DEVICE, dtype=torch.float)
test_data   = torch.tensor(test_data,   device = DEVICE, dtype=torch.float)

####################################################################################################
# STACKED AUTOENCODER MODEL
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
# LOSS FUNCTIONS
####################################################################################################

# MSE Loss function
def MSE_Loss(predicted, actual):
    # Get the mask
    mask        = actual != NORMALIZED_UNKNOWN_RATING
    mask        = mask.float()

    # Mask the columns in the output where the input is unrated
    actual      = actual    * mask
    predicted   = predicted * mask

    # Total number of ratings
    num_ratings = torch.sum(mask)

    # Calculate the square of the errors
    error       = torch.sum((actual - predicted) ** 2)
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
# TRAIN AND TEST
####################################################################################################

def train(learing_rate, weight_decay, loss_function, num_iterations, save_model=False):
        # Training on train data
    stackedAutoEncoder  = StackedAutoEncoder().to(DEVICE)
    optimizer           = optim.Adam(stackedAutoEncoder.parameters(), lr = learing_rate, weight_decay = weight_decay)

    print("Training...")
    # Train the model
    epoch_loss = []
    for i in range(num_iterations):
        predicted_ratings = stackedAutoEncoder(train_data)

        optimizer.zero_grad() 
        loss = getLoss(predicted_ratings, train_data, loss_function)
        loss.backward()
        optimizer.step()

        epoch_loss.append((i + 1, loss.data.item()))
        print("Epoch #", (i + 1), ": Training loss: ", loss.data.item())

    print("Training finished.\n")

    test_loss = test(stackedAutoEncoder, loss_function)

    if (save_model):
        print("Saving model...")
        torch.save(stackedAutoEncoder, "model")
        print("Saved model.")
    
    return epoch_loss, test_loss


def test(model, loss_function):
    print("Testing...")
    predicted_ratings = model(test_data)
    test_loss = getLoss(predicted_ratings, test_data, loss_function).data.item()
    print("Loss on test data: ", test_loss)
    print("\n")

    return test_loss

####################################################################################################
# EXPERIMENTATION
####################################################################################################

def plot_images(plot_data, labels, xlabel, ylabel, filename):
    plt.clf()
    for data, label in zip(plot_data, labels):
        xs = [x[0] for x in data]
        ys = [y[1] for y in data]
        plt.plot(xs, ys, label=label)
    plt.legend(loc='best')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.clf()

def experiment_learning_rate():
    learning_rates = [i / 100.0 for i in range(1, 10)]
    plot_data = []
    labels = []
    for learning_rate in learning_rates:
        epoch_loss, _ = train(learning_rate, WEIGHT_DECAY, "MSE", NUM_ITERATIONS, True)
        plot_data.append(epoch_loss[10:])
        labels.append("Learning rate: " + str(learning_rate))
    plot_images(plot_data, labels, "Epoch", "Mean squared error", "VaryingLearningRate.png")

def run_experiments():
    experiment_learning_rate()

####################################################################################################
# USER INTERACTION FOR TRAINING AND TESTING MODELS
####################################################################################################

mode = sys.argv[1]

if (mode == 'train'):
    # Training on train data
    train(LEARNING_RATE, WEIGHT_DECAY, LOSS_FUNCTION, NUM_ITERATIONS, True)

elif (mode == 'test'):
    # Testing on test data
    print("Loading model...")
    stackedAutoEncoder = torch.load("model")
    print("Loaded model.")

    test(stackedAutoEncoder, LOSS_FUNCTION)

elif (mode == 'exp'):
    run_experiments()

else:
    print("Usage: python3 stackedAutoencoders.py <train | test | exp>")

print('\n')

