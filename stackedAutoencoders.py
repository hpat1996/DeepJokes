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
LOSS_FUNCTION               = 'MMSE'
OPTIMIZER                   = 'Adam'
NUM_ITERATIONS              = 50
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
    def __init__(self, input_dim = num_jokes, hidden_dim = 5, output_dim = num_jokes):
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

        if ACTIVATION_FUNCTION.lower() is 'sigmoid':
            ae_seq = ae_sigmoid
        elif ACTIVATION_FUNCTION.lower() is 'relu':
            ae_seq = ae_relu
        else:
            ae_seq = ae_tanh

        if STACK_NUMBER > 0:
            self.ae01 = nn.Sequential(*ae_seq)
            self.ae02 = nn.Sequential(*ae_seq)
            self.ae03 = nn.Sequential(*ae_seq)

        if STACK_NUMBER > 3:
            self.ae04 = nn.Sequential(*ae_seq)
            self.ae05 = nn.Sequential(*ae_seq)
            self.ae06 = nn.Sequential(*ae_seq)

        if STACK_NUMBER > 6:
            self.ae07 = nn.Sequential(*ae_seq)
            self.ae08 = nn.Sequential(*ae_seq)
            self.ae09 = nn.Sequential(*ae_seq)

        if STACK_NUMBER > 9:
            self.ae10 = nn.Sequential(*ae_seq)
            self.ae11 = nn.Sequential(*ae_seq)
            self.ae12 = nn.Sequential(*ae_seq)

        if STACK_NUMBER > 12:
            self.ae13 = nn.Sequential(*ae_seq)
            self.ae14 = nn.Sequential(*ae_seq)
            self.ae15 = nn.Sequential(*ae_seq)

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
    optimizer           = optim.Adam(stackedAutoEncoder.parameters(), lr = learing_rate, weight_decay = weight_decay)

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

        epoch_train_loss.append((i + 1, loss.data.item()))
        epoch_dev_loss.append(dev(stackedAutoEncoder, loss_function))

        print("Epoch #", (i + 1), ": Training loss: ", loss.data.item())

    print("Training finished.\n")

    precision_train, recall_train = Precision_Recall_TopK(stackedAutoEncoder(train_data), train_data)
    precision_dev, recall_dev = Precision_Recall_TopK(stackedAutoEncoder(dev_data), dev_data)

    print("Precision of train data: " + str(precision_train))
    print("Recall on train data: " + str(recall_train))

    print("Precision of dev data: " + str(precision_dev))
    print("Recall on dev data: " + str(recall_dev))

    if (save_model):
        print("Saving model...")
        torch.save(stackedAutoEncoder, "model")
        print("Saved model.")

    return epoch_train_loss, epoch_dev_loss

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

def plot_dev_loss(hyper_param, dev_loss, xlabel, ylabel, filename):
    plt.clf()
    plt.plot(hyper_param, dev_loss)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.clf()

def experiment_activation_function():
    # Load a list of layer format
    # 4 of Type1: A - A - A - A - A - A
    # 12 of Type2: A - B - B - B - B - A
    # 24 Type3: A - B - B - B - B - C
    # 40 in total
    formats = [[('Sigmoid',6)                                           ],
               [('Tanh',6)                                              ],
               [('ReLU',6)                                              ],
               [('LeakyReLU',6)                                         ],
               [('Sigmoid',1),      ('Tanh',4),         ('Sigmoid',1)   ],
               [('Sigmoid',1),      ('ReLU',4),         ('Sigmoid',1)   ],
               [('Sigmoid',1),      ('LeakyReLU',4),    ('Sigmoid',1)   ],
               [('Tanh',1),         ('Sigmoid',4),      ('Tanh',1)      ],
               [('Tanh',1),         ('ReLU',4),         ('Tanh',1)      ],
               [('Tanh',1),         ('LeakyReLU',4),    ('Tanh',1)      ],
               [('ReLU',1),         ('Sigmoid',4),      ('ReLU',1)      ],
               [('ReLU',1),         ('Tanh',4),         ('ReLU',1)      ],
               [('ReLU',1),         ('LeakyReLU',4),    ('ReLU',1)      ],
               [('LeakyReLU',1),    ('Sigmoid',4),      ('LeakyReLU',1) ],
               [('LeakyReLU',1),    ('Tanh',4),         ('LeakyReLU',1) ],
               [('LeakyReLU',1),    ('ReLU',4),         ('LeakyReLU',1) ],
               [('Sigmoid',1),      ('Tanh',4),         ('ReLU',1)      ],
               [('Sigmoid',1),      ('Tanh',4),         ('LeakyReLU',1) ],
               [('Sigmoid',1),      ('ReLU',4),         ('Tanh',1)      ],
               [('Sigmoid',1),      ('ReLU',4),         ('LeakyReLU',1) ],
               [('Sigmoid',1),      ('LeakyReLU',4),    ('Tanh',1)      ],
               [('Sigmoid',1),      ('LeakyReLU',4),    ('ReLU',1)      ],
               [('Tanh',1),         ('Sigmoid',4),      ('ReLU',1)      ],
               [('Tanh',1),         ('Sigmoid',4),      ('LeakyReLU',1) ],
               [('Tanh',1),         ('ReLU',4),         ('Sigmoid',1)   ],
               [('Tanh',1),         ('ReLU',4),         ('LeakyReLU',1) ],
               [('Tanh',1),         ('LeakyReLU',4),    ('Sigmoid',1)   ],
               [('Tanh',1),         ('LeakyReLU',4),    ('ReLU',1)      ],
               [('ReLU',1),         ('Sigmoid',4),      ('Tanh',1)      ],
               [('ReLU',1),         ('Sigmoid',4),      ('LeakyReLU',1) ],
               [('ReLU',1),         ('Tanh',4),         ('Sigmoid',1)   ],
               [('ReLU',1),         ('Tanh',4),         ('LeakyReLU',1) ],
               [('ReLU',1),         ('LeakyReLU',4),    ('Sigmoid',1)   ],
               [('ReLU',1),         ('LeakyReLU',4),    ('Tanh',1)      ],
               [('LeakyReLU',1),    ('Sigmoid',4),      ('Tanh',1)      ],
               [('LeakyReLU',1),    ('Sigmoid',4),      ('ReLU',1)      ],
               [('LeakyReLU',1),    ('Tanh',4),         ('Sigmoid',1)   ],
               [('LeakyReLU',1),    ('Tanh',4),         ('ReLU',1)      ],
               [('LeakyReLU',1),    ('ReLU',4),         ('Sigmoid',1)   ],
               [('LeakyReLU',1),    ('ReLU',4),         ('Tanh',1)      ]]

    epoch_loss_list = []
    dev_loss_list = []
    precision_list = []
    recall_list = []
    for fmt in formats:
        ACTIVATION_SEQUENCE = fmt
        print("Running on " + str(ACTIVATION_SEQUENCE))
        epoch_loss, dev_loss, test_loss, precision, recall = train(LEARNING_RATE, WEIGHT_DECAY, LOSS_FUNCTION, NUM_ITERATIONS, False)
        epoch_loss_list.append(epoch_loss)
        dev_loss_list.append(dev_loss)
        precision_list.append(precision)
        recall_list.append(recall)

    # Plot loss function
    sorted_idx = np.argsort(dev_loss_list)
    plot_data = []
    labels = []
    for idx in sorted_idx[:10]:
        plot_data.append(epoch_loss_list[idx])
        labels.append(str(formats[idx]))
    plot_images(plot_data, labels, "Epoch", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingActivationFuns.png")
    with open("results/" + LOSS_FUNCTION + "_VaryingActivationFuns.txt", 'w') as f:
        for idx in sorted_idx:
            f.write('dev_lost: ' + str(dev_loss_list[idx]) + ' - format: ' + str(formats[idx]) + '\n')

    # Plot Precision
    min_precision_list = [np.min([p[1] for p in pc]) for pc in precision_list]
    sorted_idx = np.argsort(min_precision_list)
    plot_data = []
    labels = []
    for idx in sorted_idx[:10]:
        plot_data.append(precision_list[idx])
        labels.append(str(formats[idx]))
    plot_images(plot_data, labels, "Epoch", "Precision", "results/Precision_VaryingActivationFuns.png")
    with open("results/Precision_VaryingActivationFuns.txt", 'w') as f:
        for idx in sorted_idx:
            f.write('dev_lost: ' + str(min_precision_list[idx]) + ' - format: ' + str(formats[idx]) + '\n')

    # Plot Recall
    min_recall_list = [np.min([r[1] for r in rc]) for rc in recall_list]
    sorted_idx = np.argsort(min_recall_list)
    plot_data = []
    labels = []
    for idx in sorted_idx[:10]:
        plot_data.append(recall_list[idx])
        labels.append(str(formats[idx]))
    plot_images(plot_data, labels, "Epoch", "Recall", "results/Recall_VaryingActivationFuns.png")
    with open("results/Recall_VaryingActivationFuns.txt", 'w') as f:
        for idx in sorted_idx:
            f.write('dev_lost: ' + str(min_recall_list[idx]) + ' - format: ' + str(formats[idx]) + '\n')

    return formats[np.argmin(dev_loss_list)]

def experiment_optimizers():
    optimizer_list = ['adagrad', 'sgd', 'rmsprop', 'adam']
    epoch_loss_list = []
    dev_loss_list = []
    precision_list = []
    recall_list = []
    # define ACTIVATION_SEQUENCE
    for optimizer in optimizer_list:
        OPTIMIZER = optimizer
        print("Running on " + OPTIMIZER + " optimizer")
        epoch_loss, dev_loss, test_loss, precision, recall = train(LEARNING_RATE, WEIGHT_DECAY, LOSS_FUNCTION, NUM_ITERATIONS, False)
        epoch_loss_list.append(epoch_loss)
        dev_loss_list.append(dev_loss)
        precision_list.append(precision)
        recall_list.append(recall)

    # Plot loss function
    sorted_idx = np.argsort(dev_loss_list)
    plot_images(epoch_loss_list, optimizer_list, "Epoch", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingOptimizers.png")
    with open("results/" + LOSS_FUNCTION + '_VaryingOptimizers.txt', 'w') as f:
        for idx in sorted_idx:
            f.write('dev_lost: ' + str(dev_loss_list[idx]) + ' - optimizer: ' + str(optimizer_list[idx]) + '\n')

    # Plot Precision
    min_precision_list = [np.min([p[1] for p in pc]) for pc in precision_list]
    sorted_idx = np.argsort(min_precision_list)
    plot_images(precision_list, optimizer_list, "Epoch", "Precision", "results/Precision_VaryingOptimizers.png")
    with open("results/Precision_VaryingOptimizers.txt", 'w') as f:
        for idx in sorted_idx:
            f.write('dev_lost: ' + str(min_precision_list[idx]) + ' - optimizer: ' + str(optimizer_list[idx]) + '\n')

    # Plot Recall
    min_recall_list = [np.min([r[1] for r in rc]) for rc in recall_list]
    sorted_idx = np.argsort(min_recall_list)
    plot_images(recall_list, optimizer_list, "Epoch", "Recall", "results/Recall_VaryingOptimizers.png")
    with open("results/Recall_VaryingOptimizers.txt", 'w') as f:
        for idx in sorted_idx:
            f.write('dev_lost: ' + str(min_recall_list[idx]) + ' - optimizer: ' + str(optimizer_list[idx]) + '\n')

    return optimizer_list[np.argmin(dev_loss_list)]

def experiment_hidden_dimension():
    hidden_dimension_list = range(5,55,5)
    # define ACTIVATION_SEQUENCE and optimizer
    epoch_loss_list = []
    dev_loss_list = []
    precision_list = []
    recall_list = []
    label = []
    for hidden_dimension in hidden_dimension_list:
        HIDDEN_DIMENSION = hidden_dimension
        print("Running on " + str(HIDDEN_DIMENSION) + " hidden dimensions")
        epoch_loss, dev_loss, test_loss, precision, recall = train(LEARNING_RATE, WEIGHT_DECAY, LOSS_FUNCTION, NUM_ITERATIONS, False)
        epoch_loss_list.append(epoch_loss)
        dev_loss_list.append(dev_loss)
        precision_list.append(precision)
        recall_list.append(recall)
        label.append(str(hidden_dimension) + " hidden dim")

    # Plot loss function
    plot_images(epoch_loss_list, label, "Epoch", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingHiddenDim.png")
    plot_dev_loss(hidden_dimension_list, dev_loss_list, "Number of Hidden Dimensions", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingHiddenDim_Dev.png")

    # Plot Precision
    min_precision_list = [np.min([p[1] for p in pc]) for pc in precision_list]
    plot_images(precision_list, label, "Epoch", "Precision", "results/Precision_VaryingHiddenDim.png")
    plot_dev_loss(hidden_dimension_list, min_precision_list, "Number of Hidden Dimensions", "Precision", "results/Precision_VaryingHiddenDim_Dev.png")

    # Plot Recall
    min_recall_list = [np.min([r[1] for r in rc]) for rc in recall_list]
    plot_images(recall_list, label, "Epoch", "Recall", "results/Recall_VaryingHiddenDim.png")
    plot_dev_loss(hidden_dimension_list, min_recall_list, "Number of Hidden Dimensions", "Recall", "results/Recall_VaryingHiddenDim_Dev.png")

    return hidden_dimension_list[np.argmin(dev_loss_list)]

def experiment_stack_number():
    stack_number_list = range(6,54,6)
    epoch_loss_list = []
    dev_loss_list = []
    precision_list = []
    recall_list = []
    label = []
    for stack_num in stack_number_list:
        if len(ACTIVATION_SEQUENCE) is 1:
            ACTIVATION_SEQUENCE[0] = (ACTIVATION_SEQUENCE[0][0], stack_num)
        else:
            ACTIVATION_SEQUENCE[1] = (ACTIVATION_SEQUENCE[1][0], stack_num-2)
        print("Running on " + str(ACTIVATION_SEQUENCE) + " with " + str(stack_num) + " stacks")
        epoch_loss, dev_loss, test_loss, precision, recall = train(LEARNING_RATE, WEIGHT_DECAY, LOSS_FUNCTION, NUM_ITERATIONS, False)
        epoch_loss_list.append(epoch_loss)
        dev_loss_list.append(dev_loss)
        precision_list.append(precision)
        recall_list.append(recall)
        label.append("Number of stacks: " + str(stack_num))

    # Plot loss function
    plot_images(epoch_loss_list, label, "Epoch", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingStackNums.png")
    plot_dev_loss(stack_number_list, dev_loss_list, "Number of Stacks", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingStackNums_Dev.png")

    # Plot Precision
    min_precision_list = [np.min([p[1] for p in pc]) for pc in precision_list]
    plot_images(precision_list, label, "Epoch", "Precision", "results/Precision_VaryingStackNums.png")
    plot_dev_loss(stack_number_list, min_precision_list, "Number of Stacks", "Precision", "results/Precision_VaryingStackNums_Dev.png")

    # Plot Recall
    min_recall_list = [np.min([r[1] for r in rc]) for rc in recall_list]
    plot_images(recall_list, label, "Epoch", "Recall", "results/Recall_VaryingStackNums.png")
    plot_dev_loss(stack_number_list, min_recall_list, "Number of Stacks", "Recall", "results/Recall_VaryingStackNums_Dev.png")

    return stack_number_list[np.argmin(dev_loss_list)]

def experiment_learning_rate():
    learning_rates = [i / 100.0 for i in range(1, 10)]
    epoch_loss_list = []
    dev_loss_list = []
    precision_list = []
    recall_list = []
    label = []
    for learning_rate in learning_rates:
        print("Running on " + str(learning_rate) + " learning rate")
        epoch_loss, dev_loss, test_loss, precision, recall = train(learning_rate, WEIGHT_DECAY, LOSS_FUNCTION, NUM_ITERATIONS, False)
        epoch_loss_list.append(epoch_loss)
        dev_loss_list.append(dev_loss)
        precision_list.append(precision)
        recall_list.append(recall)
        label.append("Learning rate: " + str(learning_rate))

    # Plot loss function
    plot_images(epoch_loss_list, label, "Epoch", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingLearningRate.png")
    plot_dev_loss(learning_rates, dev_loss_list, "Learning Rate", LOSS_FUNCTION, "results/" + LOSS_FUNCTION + "_VaryingLearningRate_Dev.png")

    # Plot Precision
    min_precision_list = [np.min([p[1] for p in pc]) for pc in precision_list]
    plot_images(precision_list, label, "Epoch", "Precision", "results/Precision_VaryingLearningRate.png")
    plot_dev_loss(learning_rates, min_precision_list, "Learning Rate", "Precision", "results/Precision_VaryingLearningRate_Dev.png")

    # Plot Recall
    min_recall_list = [np.min([r[1] for r in rc]) for rc in recall_list]
    plot_images(recall_list, label, "Epoch", "Recall", "results/Recall_VaryingLearningRate.png")
    plot_dev_loss(learning_rates, min_recall_list, "Learning Rate", "Recall", "results/Recall_VaryingLearningRate_Dev.png")

    return learning_rates[np.argmin(dev_loss_list)]


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
    ACTIVATION_SEQUENCE = experiment_activation_function()
    # ACTIVATION_SEQUENCE = [('ReLU', 1), ('Sigmoid', 4), ('Tanh', 1)]
    OPTIMIZER = experiment_optimizers()
    # OPTIMIZER = 'sgd'
    HIDDEN_DIMENSION = experiment_hidden_dimension()
    # HIDDEN_DIMENSION = 5
    stack_num = experiment_stack_number()
    # if len(ACTIVATION_SEQUENCE) is 1:
    #     ACTIVATION_SEQUENCE[0] = (ACTIVATION_SEQUENCE[0][0], stack_num)
    # else:
    #     ACTIVATION_SEQUENCE[1] = (ACTIVATION_SEQUENCE[1][0], stack_num-2)
    learning_rate = experiment_learning_rate()
#    experiment_loss_functions()

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
