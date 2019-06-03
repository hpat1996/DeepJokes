import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.ioff()

####################################################################################################
# CONSTANTS AND HYPERPARAMETERS
####################################################################################################

# Constants for the data set
# DATASET_FILE                = "data/jester-dataset1-all.csv"
DATASET_FILE                = "data/jester-dataset1-3.csv"

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

####################################################################################################
# LOSS FUNCTIONS
####################################################################################################

# MMSE Loss function
def MMSE_Loss(predicted, actual):
    # Get the mask
    mask        = actual != NORMALIZED_UNKNOWN_RATING
    mask        = mask.astype(float)

    # Mask the columns in the output where the input is unrated
    actual      = actual    * mask
    predicted   = predicted * mask

    # Total number of ratings
    num_ratings = np.sum(mask)

    # Calculate the square of the errors
    error       = np.sum((actual - predicted) ** 2)
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
# Homework 2
####################################################################################################

def hw2_rec(R):
    # P = np.diag(np.sum(R, axis=1))
    # P_inv_half = np.diag(np.sqrt(1/np.diag(P)))
    # Q = np.diag(np.sum(R, axis=0))
    # Q_inv_half = np.diag(np.sqrt(1/np.diag(Q)))

    P_inv_half = np.diag(np.sqrt(1/np.diag(np.diag(np.sum(R, axis=1)))))
    Q_inv_half = np.diag(np.sqrt(1/np.diag(np.diag(np.sum(R, axis=0)))))

    cf_user = P_inv_half @ R @ np.transpose(R) @ P_inv_half @ R
    cf_item = R @ Q_inv_half @ np.transpose(R) @ R @ Q_inv_half

    return cf_user, cf_item

def run():
    cf_user, cf_item = hw2_rec(train_data)

    # Get Loss
    RMSE_user = getLoss(cf_user, dev_data, loss_function='RMSE')
    RMSE_item = getLoss(cf_item, dev_data, loss_function='RMSE')
    MMSE_user = getLoss(cf_user, dev_data, loss_function='MMSE')
    MMSE_item = getLoss(cf_item, dev_data, loss_function='MMSE')

    with open('results/hw2_result.txt', 'w') as f:
        f.write("User CF RMSE: " + str(RMSE_user) + "\n")
        f.write("Item CF RMSE: " + str(RMSE_item) + "\n")
        f.write("User CF MMSE: " + str(MMSE_user) + "\n")
        f.write("Item CF MMSE: " + str(MMSE_item) + "\n")

run()
