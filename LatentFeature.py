import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.random import rand

####################################################################################################
# CONSTANTS AND HYPERPARAMETERS
####################################################################################################

# Constants for the data set
DATASET_FILE                = "data/jester-dataset1-all.csv"
# DATASET_FILE                = "data/jester-dataset1-3.csv"

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

MAX_ITER                    = 100
ETA                         = 0.012
K                           = 10
LAMBDA                      = 0.1
LOSS_PER_ITER               = True
LOSS_FUNCTION               = 'MMSE'

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
# Latent Feature
####################################################################################################

def latent_feature(R):
    # Initialize values
    Es = []
    P = np.sqrt((rand(R.shape[0], K)*(NORMALIZED_MAX_RATING - NORMALIZED_MIN_RATING) + NORMALIZED_MIN_RATING)/K).astype(float)
    Q = np.sqrt((rand(R.shape[1], K)*(NORMALIZED_MAX_RATING - NORMALIZED_MIN_RATING) + NORMALIZED_MIN_RATING)/K).astype(float)

    # Run SGD
    for iter in range(MAX_ITER):
        print(str(iter) + "Iteration")
        for user_id in range(R.shape[0]):
            for movie_id in range(R.shape[1]):
                rating = R[user_id, movie_id]
                if rating is not DATASET_UNKNOWN_RATING:
                    err = 2*(rating - np.dot(P[user_id,:], Q[movie_id,:]))
                    P[user_id,:] = P[user_id,:] + ETA*(err*Q[movie_id,:] - 2*LAMBDA*P[user_id,:])
                    Q[movie_id,:] = Q[movie_id,:] + ETA*(err*P[user_id,:] - 2*LAMBDA*Q[movie_id,:])
        if LOSS_PER_ITER:
            predicted = P @ np.transpose(Q)
            E = getLoss(predicted, R, LOSS_FUNCTION)
            if LAMBDA > 0:
                for idx in range(P.shape[0]):
                    E = E + LAMBDA*np.sum(P[idx,:]*P[idx,:])
                for idx in range(Q.shape[0]):
                    E = E + LAMBDA*np.sum(Q[idx,:]*Q[idx,:])
            Es.append(E)

    if LOSS_PER_ITER:
        return P @ np.transpose(Q), Es
    else:
        return P @ np.transpose(Q)

def SVD(R):
    R = R.astype(float)
    # R[R == DATASET_UNKNOWN_RATING] = np.nan
    R[R == DATASET_UNKNOWN_RATING] = 3 # Set missing data to 0 rating (3 in normalized)
    U, S, V = np.linalg.svd(R, full_matrices=False)
    return U[:,:K] @ np.diag(S[:K]) @ V[:K,:]

def sparse_svd():
    R = R.astype(float)
    R[R == DATASET_UNKNOWN_RATING] = np.nan
    U, S, V = sp.sparse.linalg.svds(R, k=K)
    return U @ np.diag(S) @ V

####################################################################################################
# Run Code
####################################################################################################

def run_latent():
    predicted, errors = latent_feature(train_data)
    test_error = getLoss(predicted, test_data, LOSS_FUNCTION)

    print('Latent Loss function: ' + LOSS_FUNCTION + ' -> Test error: ' + str(test_error))

    with open('results/Latent_Result.txt', 'w') as f:
        f.write('Loss function: ' + LOSS_FUNCTION + ' -> Test error: ' + str(test_error))

    fig = plt.figure()
    plt.plot(range(1,MAX_ITER+1), errors)
    # fig.suptitle('Latent Feature Training')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    fig.savefig('results/Latent_Training_Errors.png')

def run_svd():
    predicted = SVD(train_data)
    test_error = getLoss(predicted, test_data, LOSS_FUNCTION)

    print('SVD Loss function: ' + LOSS_FUNCTION + ' -> Test error: ' + str(test_error))

    with open('results/SVD_Result.txt', 'w') as f:
        f.write('Loss function: ' + LOSS_FUNCTION + ' -> Test error: ' + str(test_error))

def run_sparse_svd():
    predicted = SVD(train_data)
    test_error = getLoss(predicted, test_data, LOSS_FUNCTION)

    print('Sparse SVD Loss function: ' + LOSS_FUNCTION + ' -> Test error: ' + str(test_error))

    with open('results/Sparse_SVD_Result.txt', 'w') as f:
        f.write('Loss function: ' + LOSS_FUNCTION + ' -> Test error: ' + str(test_error))

# run_latent()
run_svd()
run_sparse_svd()
