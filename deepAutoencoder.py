import time
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
DATASET_FILE                = "data/jester-dataset1-all.csv"

DATASET_UNKNOWN_RATING      = 99
DATASET_MIN_RATING          = -10
DATASET_MAX_RATING          = 10

NORMALIZED_UNKNOWN_RATING   = 99
NORMALIZED_MIN_RATING       = 1
NORMALIZED_MAX_RATING       = 5
NORMALIZED_ROUNDED          = False

NUM_DEV_TEST_USERS          = 0.5
NUM_DEV_JOKES               = 0.3
NUM_TEST_JOKES              = 0.2
NUM_DEV_TEST_JOKES          = NUM_DEV_JOKES + NUM_TEST_JOKES

# Hyperparameters for the model
ACTIVATION                  = 'ReLU'
HIDDEN_DIM                  = 75
NUM_HIDDEN_LAYERS           = 10
LEARNING_RATE               = 0.04
WEIGHT_DECAY                = 0.0
LOSS_FUNCTION               = 'MMSE'
NUM_ITERATIONS              = 100
OPTIMIZER                   = 'Adam'
MODEL_NAME                  = 'deepAutoencoder.model'

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
    
    normalized_n        = (((n - DATASET_MIN_RATING) * normalized_range) / dataset_range) + NORMALIZED_MIN_RATING

    if NORMALIZED_ROUNDED:
        normalized_n = round(normalized_n)

    return normalized_n

# Load the data from the file
# Discard first column as it is not useful
data = np.loadtxt(DATASET_FILE, dtype=np.float, delimiter=",")[:, 1:]

# Normalize the data
data = np.vectorize(normalizeData)(data)
np.random.shuffle(data)
num_users, num_jokes = data.shape

# Divide the data into train, dev and test
train_data  = np.copy(data)

dev_data    = np.zeros(shape=data.shape)
dev_data.fill(NORMALIZED_UNKNOWN_RATING)

test_data   = np.zeros(shape=data.shape)
test_data.fill(NORMALIZED_UNKNOWN_RATING)

num_dev_test_users  = int(NUM_DEV_TEST_USERS    * num_users)
num_dev_jokes       = int(NUM_DEV_JOKES         * num_jokes)
num_test_jokes      = int(NUM_TEST_JOKES        * num_jokes)
num_dev_test_jokes  = int(NUM_DEV_TEST_JOKES    * num_jokes)

train_data  [num_dev_test_users : , num_dev_test_jokes  :               ]   = NORMALIZED_UNKNOWN_RATING
dev_data    [num_dev_test_users : , -num_dev_test_jokes : -num_dev_jokes]   = data[num_dev_test_users : , -num_dev_test_jokes   : -num_dev_jokes]
test_data   [num_dev_test_users : , -num_dev_jokes      :               ]   = data[num_dev_test_users : , -num_dev_jokes        :               ]


train_data  = torch.tensor(train_data,  device = DEVICE, dtype=torch.float)
dev_data    = torch.tensor(dev_data,    device = DEVICE, dtype=torch.float)
test_data   = torch.tensor(test_data,   device = DEVICE, dtype=torch.float)

####################################################################################################
# DEEP AUTOENCODER MODEL
####################################################################################################

# The deep auto encoder model
class DeepAutoEncoder(nn.Module):
    def __init__(self, input_dim = num_jokes, hidden_dim = 60, output_dim = num_jokes, activation = 'ReLU', num_hidden_layers = 10):
        super(DeepAutoEncoder, self).__init__()

        if activation.lower() == 'relu':
            F = nn.ReLU()
        if activation.lower() == 'tanh':
            F = nn.Tanh()
        if activation.lower() == 'sigmoid':
            F = nn.Sigmoid()

        self.input_layer = nn.Linear(input_dim, hidden_dim);

        dim_reduction = 5

        self.hidden_layers_enc = nn.ModuleList([
                    nn.Sequential(nn.Linear(hidden_dim - ((i - 1) * dim_reduction), hidden_dim - (i * dim_reduction)), F)
                    for i in range(1, 1 + int(num_hidden_layers / 2), 1)])
        
        self.hidden_layers_dec = nn.ModuleList([
                    nn.Sequential(nn.Linear(hidden_dim - (i * dim_reduction), hidden_dim - ((i - 1) * dim_reduction)), F)
                    for i in range(int(num_hidden_layers / 2), 0, -1)])

        self.output_layer = nn.Linear(hidden_dim, output_dim);

    def forward(self, x):
        x = self.input_layer(x);

        for hidden_layer in self.hidden_layers_enc:
            x = hidden_layer(x);

        for hidden_layer in self.hidden_layers_dec:
            x = hidden_layer(x);

        x = self.output_layer(x);
        return x

####################################################################################################
# LOSS FUNCTIONS
####################################################################################################

def Precision_Recall_TopK(predicted, actual, K = 10):
    actual      = actual.cpu().detach().numpy()
    predicted   = predicted.cpu().detach().numpy()

    n, d        = actual.shape
    
    mask_actual = (actual    != NORMALIZED_UNKNOWN_RATING) * (actual     >= (0.6 * NORMALIZED_MAX_RATING))
    mask_pred   = (actual    != NORMALIZED_UNKNOWN_RATING) * (predicted  >= (0.6 * NORMALIZED_MAX_RATING))

    actual      = actual    * mask_actual
    predicted   = predicted * mask_pred

    precision   = 0
    recall      = 0
    for i in range(n):
        relevant_items  = set(filter(lambda item: actual[i][item] != 0, range(d)))
        topK_pred       = np.argsort(-predicted[i])[:K]
        topK_pred       = set(filter(lambda item: predicted[i][item] != 0, topK_pred))
    
        num_common  = len(relevant_items.intersection(topK_pred))
        precision   += num_common / len(topK_pred)      if len(topK_pred)       != 0    else 1
        recall      += num_common / len(relevant_items) if len(relevant_items)  != 0    else 1

    precision   = precision / n
    recall      = recall / n
    F1          = (precision * recall) / (precision + recall)
    return precision, recall, F1


# MMSE Loss function
def MMSE_Loss(predicted, actual):
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
    error, num_ratings = MMSE_Loss(predicted, actual)
    return (error / num_ratings) ** 0.5

def getLoss(predicted, actual, loss_function='MMSE'):
    if (loss_function == 'MMSE'):
        error, num_ratings = MMSE_Loss(predicted, actual)
        return error / num_ratings
    elif (loss_function == 'RMSE'):
        return RMSE_Loss(predicted, actual)

####################################################################################################
# TRAIN AND TEST
####################################################################################################

def train(hidden_dim, activation, num_hidden_layers, learing_rate, weight_decay, loss_function, num_iterations, optimizer, calculate_precision = False, save_model = False):
        # Training on train data
    deepAutoEncoder  = DeepAutoEncoder(hidden_dim = hidden_dim, activation = activation, num_hidden_layers = num_hidden_layers).to(DEVICE)

    if optimizer.lower() == 'adam':
        opt = optim.Adam(deepAutoEncoder.parameters(), lr = learing_rate, weight_decay = weight_decay)
    if optimizer.lower() == 'sgd':
        opt = optim.SGD(deepAutoEncoder.parameters(), lr = learing_rate, weight_decay = weight_decay)
    if optimizer.lower() == 'rmsprop':
        opt = optim.RMSprop(deepAutoEncoder.parameters(), lr = learing_rate, weight_decay = weight_decay)


    print("Training...")
    # Train the model
    epoch_train_loss = []
    epoch_dev_loss = []
    for i in range(num_iterations):
        predicted_ratings = deepAutoEncoder(train_data)

        opt.zero_grad() 
        loss = getLoss(predicted_ratings, train_data, loss_function)
        loss.backward()
        opt.step()

        epoch_train_loss.append((i + 1, loss.data.item()))

        dev_loss = dev(deepAutoEncoder, loss_function)
        epoch_dev_loss.append((i + 1, dev_loss))

        print("Epoch #", (i + 1), ":\t Training loss: ", round(loss.data.item(), 8), "\t Dev loss: ", round(dev_loss, 8))


    print("Training finished.\n")

    if (save_model):
        print("Saving model...")
        torch.save(deepAutoEncoder, MODEL_NAME)
        print("Saved model.")
 
    if (calculate_precision):
        precision_train,    recall_train,   F1_train    = Precision_Recall_TopK(deepAutoEncoder(train_data), train_data)
        precision_dev,      recall_dev,     F1_dev      = Precision_Recall_TopK(deepAutoEncoder(dev_data), dev_data)

        print("Precision of train data: " + str(precision_train))
        print("Recall on train data: " + str(recall_train))
        print("F1 score for train data: " + str(F1_train))
        print()

        print("Precision of dev data: " + str(precision_dev))
        print("Recall on dev data: " + str(recall_dev))
        print("F1 score for dev data: " + str(F1_dev))
        print()

        return ((epoch_train_loss, precision_train, recall_train, F1_train), (epoch_dev_loss, precision_dev, recall_dev, F1_dev))
   
    return (epoch_train_loss, epoch_dev_loss)

def dev(model, loss_function):
    predicted_ratings = model(dev_data)
    dev_loss = getLoss(predicted_ratings, dev_data, loss_function).data.item()
    return dev_loss

def test(model, loss_function):
    print("Testing...")
    predicted_ratings = model(test_data)
    test_loss = getLoss(predicted_ratings, test_data, loss_function).data.item()
    print("Loss on test data: ", test_loss)

    precision_test, recall_test = Precision_Recall_TopK(deepAutoEncoder(test_data), test_data)
    print("Precision of test data: " + str(precision_test))
    print("Recall on test data: " + str(recall_test))

    print("\n")

    return test_loss, precision_test, recall_test

####################################################################################################
# USER INTERACTION FOR TRAINING AND TESTING MODELS
####################################################################################################

mode = sys.argv[1]

if (mode == 'train'):
    # Training on train data
    start = time.time()
    train(HIDDEN_DIM, ACTIVATION, NUM_HIDDEN_LAYERS, LEARNING_RATE, WEIGHT_DECAY, LOSS_FUNCTION, NUM_ITERATIONS, OPTIMIZER, calculate_precision=False, save_model=True)
    end = time.time()
    print("Training time: " + str(round(end - start, 2)) + " seconds")

elif (mode == 'test'):
    # Testing on test data
    print("Loading model...")
    deepAutoEncoder = torch.load(MODEL_NAME)
    print("Loaded model.")

    test(deepAutoEncoder, LOSS_FUNCTION, NUM_HIDDEN_LAYERS)

else:
    print("Usage: python3 deepAutoencoders.py <train | test>")

print('\n')

