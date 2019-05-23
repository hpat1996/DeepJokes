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

NUM_DEV_TEST_USERS          = 0.5
NUM_DEV_JOKES               = 0.3
NUM_TEST_JOKES              = 0.2
NUM_DEV_TEST_JOKES          = NUM_DEV_JOKES + NUM_TEST_JOKES

# Hyperparameters for the model
LEARNING_RATE               = 0.04
WEIGHT_DECAY                = 0.0
LOSS_FUNCTION               = 'RMSE'
OPTIMIZER                   = 'Adam'
NUM_ITERATIONS              = 100
ACTIVATION_FUNCTION         = 'Tanh'
HIDDEN_DIMENSION            = 5
STACK_NUMBER                = 6

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
# STACKED AUTOENCODER MODEL
####################################################################################################

# The stacked auto encoder model
class StackedAutoEncoder(nn.Module):
    def __init__(self, input_dim = num_jokes, hidden_dim = HIDDEN_DIMENSION, output_dim = num_jokes):
        super(StackedAutoEncoder, self).__init__()

        ae_sigmoid = [
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
        ]

        ae_tanh = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        ]

        ae_relu = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ]

        if ACTIVATION_FUNCTION.lower() == 'sigmoid':
            ae_seq = ae_sigmoid
        elif ACTIVATION_FUNCTION.lower() == 'relu':
            ae_seq = ae_relu
        else:
            ae_seq = ae_tanh

        if STACK_NUMBER > 0:
            self.ae01 = nn.Sequential(*ae_seq.copy())
            self.ae02 = nn.Sequential(*ae_seq.copy())
            self.ae03 = nn.Sequential(*ae_seq.copy())

        if STACK_NUMBER > 3:
            self.ae04 = nn.Sequential(*ae_seq.copy())
            self.ae05 = nn.Sequential(*ae_seq.copy())
            self.ae06 = nn.Sequential(*ae_seq.copy())

        if STACK_NUMBER > 6:
            self.ae07 = nn.Sequential(*ae_seq.copy())
            self.ae08 = nn.Sequential(*ae_seq.copy())
            self.ae09 = nn.Sequential(*ae_seq.copy())

        if STACK_NUMBER > 9:
            self.ae10 = nn.Sequential(*ae_seq.copy())
            self.ae11 = nn.Sequential(*ae_seq.copy())
            self.ae12 = nn.Sequential(*ae_seq.copy())

        if STACK_NUMBER > 12:
            self.ae13 = nn.Sequential(*ae_seq.copy())
            self.ae14 = nn.Sequential(*ae_seq.copy())
            self.ae15 = nn.Sequential(*ae_seq.copy())

    def forward(self, x):
        if STACK_NUMBER > 0:
            x = self.ae01(x)
            x = self.ae02(x)
            x = self.ae03(x)

        if STACK_NUMBER > 3:
            x = self.ae04(x)
            x = self.ae05(x)
            x = self.ae06(x)

        if STACK_NUMBER > 6:
            x = self.ae07(x)
            x = self.ae08(x)
            x = self.ae09(x)

        if STACK_NUMBER > 9:
            x = self.ae10(x)
            x = self.ae11(x)
            x = self.ae12(x)

        if STACK_NUMBER > 12:
            x = self.ae13(x)
            x = self.ae14(x)
            x = self.ae15(x)

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

    return ((precision / n), (recall / n))


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

def train(learing_rate, weight_decay, loss_function, num_iterations, save_model=False):
        # Training on train data
    stackedAutoEncoder  = StackedAutoEncoder().to(DEVICE)
    # optimizer           = optim.Adam(stackedAutoEncoder.parameters(), lr = learing_rate, weight_decay = weight_decay)
    if OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(stackedAutoEncoder.parameters(), lr = learing_rate, weight_decay = weight_decay, momentum = 0.9)
    elif OPTIMIZER.lower() == 'rmsprop':
        optimizer = optim.RMSprop(stackedAutoEncoder.parameters(), lr = learing_rate, weight_decay = weight_decay, momentum = 0.9)
    else: # Adam(default)
        optimizer = optim.Adam(stackedAutoEncoder.parameters(), lr = learing_rate, weight_decay = weight_decay)

    print("Training...")
    # Train the model
    epoch_train_loss = []
    epoch_dev_loss = []
    for i in range(num_iterations):
        predicted_ratings = stackedAutoEncoder(train_data)

        optimizer.zero_grad()
        loss = getLoss(predicted_ratings, train_data, loss_function)
        loss.backward()
        optimizer.step()

        dev_loss = dev(stackedAutoEncoder, loss_function)

        epoch_train_loss.append((i + 1, loss.data.item()))
        epoch_dev_loss.append((i + 1, dev_loss))

        print("Epoch #", (i + 1), ": Training loss: ", loss.data.item())
        print("Epoch #", (i + 1), ": Dev loss: ", dev_loss)

    print("Training finished.\n")

    precision_train, recall_train = Precision_Recall_TopK(stackedAutoEncoder(train_data), train_data)
    precision_dev, recall_dev = Precision_Recall_TopK(stackedAutoEncoder(dev_data), dev_data)

    print("Precision of train data: " + str(precision_train))
    print("Recall on train data: " + str(recall_train))

    print("Precision of dev data: " + str(precision_dev))
    print("Recall on dev data: " + str(recall_dev))

    print("\n")

    if (save_model):
        print("Saving model...")
        torch.save(stackedAutoEncoder, "model")
        print("Saved model.")
        print("\n")

    return epoch_train_loss, epoch_dev_loss, precision_dev, recall_dev

def dev(model, loss_function):
    predicted_ratings = model(dev_data)
    dev_loss = getLoss(predicted_ratings, dev_data, loss_function).data.item()
    return dev_loss

def test(model, loss_function):
    print("Testing...")
    predicted_ratings = model(test_data)
    test_loss = getLoss(predicted_ratings, test_data, loss_function).data.item()
    print("Loss on test data: ", test_loss)

    precision_test, recall_test = Precision_Recall_TopK(stackedAutoEncoder(test_data), test_data)
    print("Precision of test data: " + str(precision_test))
    print("Recall on test data: " + str(recall_test))

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

def experiment_optimizers():
    train_loss_list = []
    dev_loss_list = []
    precision_list = []
    recall_list = []
    optimizer_list = ['SGD', 'RMSprop', 'Adam']
    for optimizer in optimizer_list:
        OPTIMIZER = optimizer
        print("Running on " + str(OPTIMIZER))
        epoch_train_loss, epoch_dev_loss, precision_dev, recall_dev = train(LEARNING_RATE, WEIGHT_DECAY, LOSS_FUNCTION, NUM_ITERATIONS, False)
        train_loss_list.append(epoch_train_loss)
        dev_loss_list.append(epoch_dev_loss)
        precision_list.append(precision_dev)
        recall_list.append(recall_dev)

    # Plot loss function
    plot_images(train_loss_list, optimizer_list, "Epoch", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingOptimzers_train.png")
    plot_images(dev_loss_list, optimizer_list, "Epoch", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingOptimzers_dev.png")
    with open("results/" + LOSS_FUNCTION + "_VaryingOptimzers.txt", 'w') as f:
        for idx in range(len(optimizer_list)):
            f.write('Optimizer: ' + str(optimizer_list[idx]) + '\t' + LOSS_FUNCTION + ' Train: ' + str(train_loss_list[idx][-1]) + '\n')
        f.write('\n')
        for idx in range(len(optimizer_list)):
            f.write('Optimizer: ' + str(optimizer_list[idx]) + '\t' + LOSS_FUNCTION + ' Dev: ' + str(dev_loss_list[idx][-1]) + '\n')
        f.write('\n')
        for idx in range(len(optimizer_list)):
            f.write('Optimizer: ' + str(optimizer_list[idx]) + '\t Precision: ' + str(precision_list[idx]) + '\n')
        f.write('\n')
        for idx in range(len(optimizer_list)):
            f.write('Optimizer: ' + str(optimizer_list[idx]) + '\t Recall: ' + str(recall_list[idx]) + '\n')

    return optimizer_list[np.argmin([l[-1][1] for l in dev_loss_list])]

def experiment_activation_function():
    train_loss_list = []
    dev_loss_list = []
    precision_list = []
    recall_list = []
    func_type_list = ['Sigmoid', 'ReLU', 'Tanh']
    for func_type in func_type_list:
        ACTIVATION_FUNCTION = func_type
        print("Running on " + str(ACTIVATION_FUNCTION))
        epoch_train_loss, epoch_dev_loss, precision_dev, recall_dev = train(LEARNING_RATE, WEIGHT_DECAY, LOSS_FUNCTION, NUM_ITERATIONS, False)
        train_loss_list.append(epoch_train_loss)
        dev_loss_list.append(epoch_dev_loss)
        precision_list.append(precision_dev)
        recall_list.append(recall_dev)

    # Plot loss function
    plot_images(train_loss_list, func_type_list, "Epoch", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingActivationFuns_train.png")
    plot_images(dev_loss_list, func_type_list, "Epoch", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingActivationFuns_dev.png")
    with open("results/" + LOSS_FUNCTION + "_VaryingActivationFuns.txt", 'w') as f:
        for idx in range(len(func_type_list)):
            f.write('Activation Function: ' + str(func_type_list[idx]) + '\t' + LOSS_FUNCTION + ' Train: ' + str(train_loss_list[idx][-1]) + '\n')
        f.write('\n')
        for idx in range(len(func_type_list)):
            f.write('Activation Function: ' + str(func_type_list[idx]) + '\t' + LOSS_FUNCTION + ' Dev: ' + str(dev_loss_list[idx][-1]) + '\n')
        f.write('\n')
        for idx in range(len(func_type_list)):
            f.write('Activation Function: ' + str(func_type_list[idx]) + '\t Precision: ' + str(precision_list[idx]) + '\n')
        f.write('\n')
        for idx in range(len(func_type_list)):
            f.write('Activation Function: ' + str(func_type_list[idx]) + '\t Recall: ' + str(recall_list[idx]) + '\n')

    return func_type_list[np.argmin([l[-1][1] for l in dev_loss_list])]

def experiment_hidden_dimension():
    train_loss_list = []
    dev_loss_list = []
    precision_list = []
    recall_list = []
    label = []
    hidden_dimension_list = [4, 8, 16]
    for hidden_dimension in hidden_dimension_list:
        HIDDEN_DIMENSION = hidden_dimension
        print("Running on " + "Num of Hidden Dims: " +  str(HIDDEN_DIMENSION))
        epoch_train_loss, epoch_dev_loss, precision_dev, recall_dev = train(LEARNING_RATE, WEIGHT_DECAY, LOSS_FUNCTION, NUM_ITERATIONS, False)
        train_loss_list.append(epoch_train_loss)
        dev_loss_list.append(epoch_dev_loss)
        precision_list.append(precision_dev)
        recall_list.append(recall_dev)
        label.append("Num of Hidden Dims: " + str(HIDDEN_DIMENSION))

    # Plot loss function
    plot_images(train_loss_list, label, "Epoch", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingHiddemDims_train.png")
    plot_images(dev_loss_list, label, "Epoch", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingHiddemDims_dev.png")
    with open("results/" + LOSS_FUNCTION + "_VaryingHiddemDims.txt", 'w') as f:
        for idx in range(len(hidden_dimension_list)):
            f.write('Number of Hiddem Dimensions: ' + str(hidden_dimension_list[idx]) + '\t' + LOSS_FUNCTION + ' Train: ' + str(train_loss_list[idx][-1]) + '\n')
        f.write('\n')
        for idx in range(len(hidden_dimension_list)):
            f.write('Number of Hiddem Dimensions: ' + str(hidden_dimension_list[idx]) + '\t' + LOSS_FUNCTION + ' Dev: ' + str(dev_loss_list[idx][-1]) + '\n')
        f.write('\n')
        for idx in range(len(hidden_dimension_list)):
            f.write('Number of Hiddem Dimensions: ' + str(hidden_dimension_list[idx]) + '\t Precision: ' + str(precision_list[idx]) + '\n')
        f.write('\n')
        for idx in range(len(hidden_dimension_list)):
            f.write('Number of Hiddem Dimensions: ' + str(hidden_dimension_list[idx]) + '\t Recall: ' + str(recall_list[idx]) + '\n')

    return hidden_dimension_list[np.argmin([l[-1][1] for l in dev_loss_list])]

def experiment_stack_number():
    train_loss_list = []
    dev_loss_list = []
    precision_list = []
    recall_list = []
    label = []
    stack_num_list = [3, 6, 9, 12, 15]
    for stack_num in stack_num_list:
        STACK_NUMBER = stack_num
        print("Running on " + "Num of Stacks: " + str(STACK_NUMBER))
        epoch_train_loss, epoch_dev_loss, precision_dev, recall_dev = train(LEARNING_RATE, WEIGHT_DECAY, LOSS_FUNCTION, NUM_ITERATIONS, False)
        train_loss_list.append(epoch_train_loss)
        dev_loss_list.append(epoch_dev_loss)
        precision_list.append(precision_dev)
        recall_list.append(recall_dev)
        label.append("Num of Stacks: " + str(STACK_NUMBER))

    # Plot loss function
    plot_images(train_loss_list, label, "Epoch", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingStacks_train.png")
    plot_images(dev_loss_list, label, "Epoch", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingStacks_dev.png")
    with open("results/" + LOSS_FUNCTION + "_VaryingStacks.txt", 'w') as f:
        for idx in range(len(stack_num_list)):
            f.write('Number of Stacks: ' + str(stack_num_list[idx]) + '\t' + LOSS_FUNCTION + ' Train: ' + str(train_loss_list[idx][-1]) + '\n')
        f.write('\n')
        for idx in range(len(stack_num_list)):
            f.write('Number of Stacks: ' + str(stack_num_list[idx]) + '\t' + LOSS_FUNCTION + ' Dev: ' + str(dev_loss_list[idx][-1]) + '\n')
        f.write('\n')
        for idx in range(len(stack_num_list)):
            f.write('Number of Stacks: ' + str(stack_num_list[idx]) + '\t Precision: ' + str(precision_list[idx]) + '\n')
        f.write('\n')
        for idx in range(len(stack_num_list)):
            f.write('Number of Stacks: ' + str(stack_num_list[idx]) + '\t Recall: ' + str(recall_list[idx]) + '\n')

    return stack_num_list[np.argmin([l[-1][1] for l in dev_loss_list])]

def experiment_learning_rate():
    train_loss_list = []
    dev_loss_list = []
    precision_list = []
    recall_list = []
    label = []
    learning_rate_list = [i / 100.0 for i in range(1, 10)]
    for learning_rate in learning_rate_list:
        print("Running on " + "Learning rate: " + str(learning_rate))
        epoch_train_loss, epoch_dev_loss, precision_dev, recall_dev = train(LEARNING_RATE, WEIGHT_DECAY, LOSS_FUNCTION, NUM_ITERATIONS, False)
        train_loss_list.append(epoch_train_loss)
        dev_loss_list.append(epoch_dev_loss)
        precision_list.append(precision_dev)
        recall_list.append(recall_dev)
        label.append("Learning rate: " + str(learning_rate))

    # Plot loss function
    plot_images(train_loss_list, label, "Epoch", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingLearnigRates_train.png")
    plot_images(dev_loss_list, label, "Epoch", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingLearnigRates_dev.png")
    with open("results/" + LOSS_FUNCTION + "_VaryingLearnigRates.txt", 'w') as f:
        for idx in range(len(learning_rate_list)):
            f.write('Learning Rate: ' + str(learning_rate_list[idx]) + '\t' + LOSS_FUNCTION + ' Train: ' + str(train_loss_list[idx][-1]) + '\n')
        f.write('\n')
        for idx in range(len(learning_rate_list)):
            f.write('Learning Rate: ' + str(learning_rate_list[idx]) + '\t' + LOSS_FUNCTION + ' Dev: ' + str(dev_loss_list[idx][-1]) + '\n')
        f.write('\n')
        for idx in range(len(learning_rate_list)):
            f.write('Learning Rate: ' + str(learning_rate_list[idx]) + '\t Precision: ' + str(precision_list[idx]) + '\n')
        f.write('\n')
        for idx in range(len(learning_rate_list)):
            f.write('Learning Rate: ' + str(learning_rate_list[idx]) + '\t Recall: ' + str(recall_list[idx]) + '\n')

    return learning_rate_list[np.argmin([l[-1][1] for l in dev_loss_list])]

def experiment_loss_functions():
    learning_rate = 0.01
    plot_data = []
    labels = []

    epoch_loss, _ = train(learning_rate, WEIGHT_DECAY, "MMSE", NUM_ITERATIONS, True)
    plot_data.append(epoch_loss[10:])
    labels.append("MMSE")

    epoch_loss, _ = train(learning_rate, WEIGHT_DECAY, "RMSE", NUM_ITERATIONS, True)
    plot_data.append(epoch_loss[10:])
    labels.append("RMSE")

    plot_images(plot_data, labels, "Epoch", "Error", "VaryingLossFunction.png")


def run_experiments():
    OPTIMIZER = experiment_optimizers()
    # OPTIMIZER = 'sgd'
    ACTIVATION_FUNCTION = experiment_activation_function()
    # ACTIVATION_FUNCTION = [('ReLU', 1), ('Sigmoid', 4), ('Tanh', 1)]
    HIDDEN_DIMENSION = experiment_hidden_dimension()
    # HIDDEN_DIMENSION = 5
    STACK_NUMBER = experiment_stack_number()
    # STACK_NUMBER = 6
    LEARNING_RATE = experiment_learning_rate()
    # LEARNING_RATE = 0.4
    # experiment_loss_functions()

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
    # Run the experiments
    run_experiments()

else:
    print("Usage: python3 stackedAutoencoders.py <train | test | exp>")

print('\n')
