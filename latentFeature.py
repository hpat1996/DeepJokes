import math
import random
import time
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
DATASET_FILE                = "data/jester-dataset1-all.csv"

DATASET_UNKNOWN_RATING      = 99
DATASET_MIN_RATING          = -10
DATASET_MAX_RATING          = 10

NORMALIZED_UNKNOWN_RATING   = 0
NORMALIZED_MIN_RATING       = 1
NORMALIZED_MAX_RATING       = 5
NORMALIZED_ROUNDED          = False

NUM_DEV_TEST_USERS          = 0.5
NUM_DEV_JOKES               = 0.3
NUM_TEST_JOKES              = 0.2
NUM_DEV_TEST_JOKES          = NUM_DEV_JOKES + NUM_TEST_JOKES

# Hyperparameters for the model
NUM_LATENT_FEATURES         = 10
NUM_ITERATIONS              = 100
LEARNING_RATE               = 1.0
 

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

mask = data != NORMALIZED_UNKNOWN_RATING
num_rated = np.sum(mask)

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

def Precision_Recall_TopK(predicted, actual, K = 10):
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
    F1          = 2 * (precision * recall) / (precision + recall)
    return precision, recall, F1

def RMSE(predicted, actual):
    mask        = actual != NORMALIZED_UNKNOWN_RATING

    actual      = actual    * mask
    predicted   = predicted * mask

    num_ratings = np.sum(mask)
    error       = np.sum((actual - predicted) ** 2)
    
    return (error / num_ratings) ** 0.5



####################################################################################################
# TRAIN AND TEST
####################################################################################################
def getRatings(data):
    R = []
    for i in range(num_users):
        for j in range(num_jokes):
            if data[i, j] != NORMALIZED_UNKNOWN_RATING:
                R.append((i, j, data[i, j]))
    random.shuffle(R)
    return R

def train(num_latent_features, num_iterations, learning_rate, calculate_precision = False):
    # Training on train data
    P = np.random.uniform(low = 0, high = math.sqrt(NORMALIZED_MAX_RATING / num_latent_features), size=(num_users, num_latent_features))
    Q = np.random.uniform(low = 0, high = math.sqrt(NORMALIZED_MAX_RATING / num_latent_features), size=(num_jokes, num_latent_features))

    print("Training...")
    
    epoch_train_loss    = []
    epoch_dev_loss      = []
    time_train_loss     = []
    time_dev_loss       = []
    start_time          = time.time()

    for i in range(num_iterations):

        P_copy = np.copy(P)
        Q_copy = np.copy(Q)

        for user in range(num_users):
            actual_ratings      = train_data[user]
            predicted_ratings   = np.matmul(P[user], Q.T)
            
            mask                = actual_ratings != NORMALIZED_UNKNOWN_RATING
            actual_ratings      = actual_ratings * mask
            predicted_ratings   = predicted_ratings * mask
             
            diff                = (actual_ratings - predicted_ratings) 
            err                 = (2 * np.sum(diff)) / num_rated
            
            P_copy[user]        += 2 * learning_rate * (np.matmul(diff, Q)) / num_rated
            Q_copy              += 2 * learning_rate * (np.matmul(diff.T.reshape(num_jokes, 1), P[user].T.reshape(1, num_latent_features))) / num_rated
        
        P = np.copy(P_copy)
        Q = np.copy(Q_copy)


        end_time = time.time() - start_time

        predicted_ratings   = np.matmul(P, Q.T)
        train_loss          = RMSE(predicted_ratings, train_data)
        dev_loss            = RMSE(predicted_ratings, dev_data)

        epoch_train_loss.append((i + 1, train_loss))
        time_train_loss.append((end_time, train_loss))

        epoch_dev_loss.append((i + 1, dev_loss))
        time_dev_loss.append((end_time, dev_loss))

        print("Epoch #", (i + 1), ":\t Training loss: ", round(train_loss, 8), "\t Dev loss: ", round(dev_loss, 8))


    print("Training finished.\n")

    if (calculate_precision):
        predicted_ratings = np.matmul(P, Q.T)
        precision_train,    recall_train,   F1_train    = Precision_Recall_TopK(predicted_ratings, train_data)
        precision_dev,      recall_dev,     F1_dev      = Precision_Recall_TopK(predicted_ratings, dev_data)

        print("Precision of train data: "   + str(precision_train))
        print("Recall on train data: "      + str(recall_train))
        print("F1 score for train data: "   + str(F1_train))
        print()

        print("Precision of dev data: "     + str(precision_dev))
        print("Recall on dev data: "        + str(recall_dev))
        print("F1 score for dev data: "     + str(F1_dev))
        print()

        train_metrics   = (epoch_train_loss, time_train_loss, precision_train, recall_train, F1_train)
        dev_metrics     = (epoch_dev_loss, time_dev_loss, precision_dev, recall_dev, F1_dev)
        return (P, Q, train_metrics, dev_metrics)
   
    train_metrics   = (epoch_train_loss, time_train_loss)
    dev_metrics     = (epoch_dev_loss, time_dev_loss)
    return (P, Q, train_metrics, dev_metrics)

def test(P, Q):
    print("Testing...")
    predicted_ratings   = np.matmul(P, Q.T)
    test_loss           = RMSE(predicted_ratings, test_data)
    print("Loss on test data: ", test_loss)

    precision_test, recall_test, f1_test = Precision_Recall_TopK(predicted_ratings, test_data)
    print("Precision of test data: " + str(precision_test))
    print("Recall on test data: " + str(recall_test))
    print("F1 on test data: " + str(f1_test))

    print("\n")

    return test_loss, precision_test, recall_test

####################################################################################################
# PLOT
####################################################################################################

def plot_images(plot_data, labels, xlabel, ylabel, filename):
    refined_data = []
    for data in plot_data:
        refined_data.append(list(filter(lambda x: x[1] < 25, data)))

    plt.clf()
    for data, label in zip(refined_data, labels):
        xs = [x[0] for x in data]
        ys = [y[1] for y in data]
        plt.plot(xs, ys, label=label)
    plt.legend(loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.clf()

####################################################################################################
# USER INTERACTION FOR TRAINING AND TESTING MODELS
####################################################################################################

mode = sys.argv[1]

if (mode == 'train'):
    # Training on train data
    plot_data_train     = []
    plot_data_dev       = []
    time_data_train     = []
    time_data_dev       = []
    labels = []

    P, Q, train_metrics, dev_metrics = train(NUM_LATENT_FEATURES, NUM_ITERATIONS, LEARNING_RATE, calculate_precision = True)
 
    plot_data_train.append(train_metrics[0])
    plot_data_dev.append(dev_metrics[0])
    time_data_train.append(train_metrics[1])
    time_data_dev.append(dev_metrics[1])
    label = "Latent Feature recommendation"
    labels.append(label)

    plot_images(plot_data_train, labels, "Epoch", "Squared error", "images/LatentFeature_RMSE_Train.png")
    plot_images(plot_data_dev, labels, "Epoch", "Squared error", "images/LatentFeature_RMSE_Dev.png")
    plot_images(time_data_train, labels, "Time", "Squared error", "images/LatentFeature_RMSE_Train_Timed.png")
    plot_images(time_data_dev, labels, "Time", "Squared error", "images/LatentFeature_RMSE_Dev_Timed.png")

    test(P, Q)

    with open("images/Time_Error.txt", "a") as f:
        f.write(','.join(str(data[0]) for data in train_metrics[1]))
        f.write('\n')
        f.write(','.join(str(data[1]) for data in train_metrics[1]))
        f.write('\n')
        f.write(','.join(str(data[0]) for data in dev_metrics[1]))
        f.write('\n')
        f.write(','.join(str(data[1]) for data in dev_metrics[1]))
        f.write('\n')

else:
    print("Usage: python3 latentFeature.py train")

print('\n')


