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

                            # (% users(%train, %dev), % jokes (%train))
NUM_TRAIN                   = ((0.5, 0.3), 0.5)

# Hyperparameters for the model
LEARNING_RATE               = 0.04
WEIGHT_DECAY                = 0.0
LOSS_FUNCTION               = 'RMSE'
OPTIMIZER                   = 'Adam'
NUM_ITERATIONS              = 100
ACTIVATION_FORMAT           = [('Sigmoid',1), ('Tanh',4), ('ReLU', 1)]
HIDDEN_DIMENSION            = 5
    
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
np.random.shuffle(data)
num_users, num_jokes = data.shape

# Divide the data into train and test
num_train_users   = int(NUM_TRAIN[0][0] * num_users)
num_dev_users     = int((NUM_TRAIN[0][0] + NUM_TRAIN[0][1]) * num_users)
num_train_jokes   = int(NUM_TRAIN[1] * num_jokes)

train_data  = np.zeros(shape=data.shape)
train_data.fill(NORMALIZED_UNKNOWN_RATING)
dev_data    = np.zeros(shape=data.shape)
dev_data.fill(NORMALIZED_UNKNOWN_RATING)
test_data   = np.zeros(shape=data.shape)
test_data.fill(NORMALIZED_UNKNOWN_RATING)

train_data  [               :               ,                       :num_train_jokes]   = data[               :               ,                       :num_train_jokes]
train_data  [               :num_train_users,                       :               ]   = data[               :num_train_users,                       :               ]
dev_data    [num_train_users:num_dev_users  ,   num_train_jokes     :               ]   = data[num_train_users:num_dev_users  ,   num_train_jokes     :               ]
test_data   [num_dev_users  :               ,   num_train_jokes     :               ]   = data[num_dev_users  :               ,   num_train_jokes     :               ]


train_data  = torch.tensor(train_data,  device = DEVICE, dtype=torch.float)
dev_data    = torch.tensor(dev_data,  device = DEVICE, dtype=torch.float)
test_data   = torch.tensor(test_data,   device = DEVICE, dtype=torch.float)

####################################################################################################
# STACKED AUTOENCODER MODEL
####################################################################################################

# The stacked auto encoder model
# ACTIVATION_ORDER is a list of tuple: (type of activation function, number of iterations)
class StackedAutoEncoder(nn.Module):
    def __init__(self, input_dim = num_jokes, hidden_dim = HIDDEN_DIMENSION, output_dim = num_jokes, activation_format = ACTIVATION_FORMAT):
        super(StackedAutoEncoder, self).__init__()

        self.format = activation_format

        self.Sigmoid = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.Tanh = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.ReLU = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.LeakyReLU = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    # Run forward as given in ACTIVATION_ORDER
    def forward(self, x):
        for fmt in self.format:
            for idx in range(fmt[1]):
                if fmt[0].lower() == 'sigmoid':
                    x = self.Sigmoid(x)
                elif fmt[0].lower() == 'tanh':
                    x = self.Tanh(x)
                elif fmt[0].lower() == 'relu':
                    x = self.ReLU(x)
                elif fmt[0].lower() == 'leakyrelu':
                    x = self.LeakyReLU(x)

        return x

####################################################################################################
# LOSS FUNCTIONS
####################################################################################################

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

# MAE (Mean Absolute Error) Loss Function
def MAE_Loss(predicted, actual):
    # Get the mask
    mask        = actual != NORMALIZED_UNKNOWN_RATING
    mask        = mask.float()

    # Mask the columns in the output where the input is unrated
    actual      = actual    * mask
    predicted   = predicted * mask

    # Total number of ratings
    num_ratings = torch.sum(mask)

    # Calculate the square of the errors
    error       = torch.sum(torch.abs(actual - predicted))
    return error, num_ratings

def Precision_Recall(predicted, actual):
    actual_cloned = actual.detach().numpy()
    predicted_cloned = predicted.detach().numpy()
    precision = 0.0
    recall = 0.0
    for i in range(0, actual_cloned.shape[0]):
        relevant_items = set(filter(lambda j : actual_cloned[i][j] >= 3.0, np.arange(0, actual_cloned.shape[1])))
        top_k_rec = np.argsort(-predicted_cloned[i])[:10]
        top_k_rec_filtered = set(filter(lambda j : predicted_cloned[i][j] >= 3.0,  top_k_rec))
        length1 = len(top_k_rec_filtered)
        length2 = len(relevant_items)
        val = len(relevant_items.intersection(top_k_rec_filtered))
        if length1 == 0:
            precision += 1
        else:
            precision += val/float(length1)
        if length2 == 0:
            recall += 1
        else:
            recall += val/float(length2)
    return ((precision/actual_cloned.shape[0], recall/actual_cloned.shape[0]))

def getLoss(predicted, actual, loss_function='MMSE'):
    if (loss_function == 'MMSE'):
        error, num_ratings = MMSE_Loss(predicted, actual)
        return error / num_ratings
    elif (loss_function == 'RMSE'):
        return RMSE_Loss(predicted, actual)
    elif (loss_function == 'MAE'):
        return MAE_Loss(predicted, actual)
    elif (loss_function == 'precision'):
        return Precision_Recall(predicted, actual)[0]
    elif (loss_function == 'recall'):
        return Precision_Recall(predicted, actual)[1]

####################################################################################################
# TRAIN AND TEST
####################################################################################################

def dev(model, loss_function):
    print("Validating...")
    predicted_ratings = model(dev_data)
    dev_loss = getLoss(predicted_ratings, dev_data, loss_function).data.item()
    print("Loss on dev data: ", dev_loss)
    print("\n")

    return dev_loss

def test(model, loss_function):
    print("Testing...")
    predicted_ratings = model(test_data)
    test_loss = getLoss(predicted_ratings, test_data, loss_function).data.item()
    print("Loss on test data: ", test_loss)
    print("\n")

    return test_loss

def train(learing_rate, weight_decay, loss_function, num_iterations, save_model=False):
        # Training on train data
    stackedAutoEncoder  = StackedAutoEncoder().to(DEVICE)
    if OPTIMIZER.lower() == 'adagrad':
        optimizer = optim.Adagrad(stackedAutoEncoder.parameters(), lr = learing_rate, weight_decay = weight_decay)
    elif OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(stackedAutoEncoder.parameters(), lr = learing_rate, weight_decay = weight_decay, momentum = 0.9)
    elif OPTIMIZER.lower() == 'rmsprop':
        optimizer = optim.RMSprop(stackedAutoEncoder.parameters(), lr = learing_rate, weight_decay = weight_decay, momentum = 0.9)
    else: # Adam(default)
        optimizer = optim.Adam(stackedAutoEncoder.parameters(), lr = learing_rate, weight_decay = weight_decay)

    print("Training...")
    # Train the model
    epoch_loss = []
    precisions = []
    recalls    = []
    
    for i in range(num_iterations):
        predicted_ratings = stackedAutoEncoder(train_data)

        optimizer.zero_grad()
        loss = getLoss(predicted_ratings, train_data, loss_function)
        precision, recall = Precision_Recall(predicted_ratings, train_data)
        loss.backward()
        optimizer.step()

        epoch_loss.append((i + 1, loss.data.item()))
        precisions.append((i + 1, precision))
        recalls.append((i + 1, recall))
        print("Epoch #", (i + 1), ": Training loss: ", loss.data.item())
        print("Epoch #", (i + 1), ": Precision: ", precision)
        print("Epoch #", (i + 1), ": Recall: ", recall)
    print("Training finished.\n")
    
    dev_loss = dev(stackedAutoEncoder, loss_function)
    dev_precision = dev(stackedAutoEncoder, 'precision')
    dev_recall = dev(stackedAutoEncoder, 'recall')
    test_loss = test(stackedAutoEncoder, loss_function)

    if (save_model):
        print("Saving model...")
        torch.save(stackedAutoEncoder, "model")
        print("Saved model.")

    return epoch_loss, dev_loss, test_loss, precisions, dev_precision, recalls, dev_recall

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
               [('Tanh',1),         ('Sigmoid',4),      ('ReLU',1)]     ,
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
    dev_precision_list = []
    recall_list = []
    dev_recall_list = []
    for fmt in formats:
        ACTIVATION_FORMAT = fmt
        print("Running on " + str(ACTIVATION_FORMAT))
        epoch_loss, dev_loss, test_loss, precision, dev_precision, recall, dev_recall = train(LEARNING_RATE, WEIGHT_DECAY, LOSS_FUNCTION, NUM_ITERATIONS, False)
        epoch_loss_list.append(epoch_loss)
        dev_loss_list.append(dev_loss)
        precision_list.append(precision)
        dev_precision_list.append(dev_precision)
        recall_list.append(recall)
        dev_recall_list.append(dev_recall)
        
    # Plot loss function
    sorted_idx = np.argsort(dev_loss_list)
    plot_data = []
    labels = []
    for idx in sorted_idx[:10]:
        plot_data.append(epoch_loss_list[idx])
        labels.append(str(formats[idx]))
    plot_images(plot_data, labels, "Epoch", LOSS_FUNCTION, LOSS_FUNCTION + "_VaryingActivationFuns.png")
    with open(LOSS_FUNCTION + '_VaryingActivationFuns.txt', 'w') as f:
        for idx in sorted_idx:
            f.write('dev_lost: ' + dev_loss_list[idx] + ' - format: ' + str(formats[idx]) + '\n')
    
    # Plot Precision
    sorted_idx = np.argsort(dev_precision_list)
    plot_data = []
    labels = []
    for idx in sorted_idx[:10]:
        plot_data.append(precision_list[idx])
        labels.append(str(formats[idx]))
    plot_images(plot_data, labels, "Epoch", "Precision", "Precision_VaryingActivationFuns.png")
    with open('Precision_VaryingActivationFuns.txt', 'w') as f:
        for idx in sorted_idx:
            f.write('dev_lost: ' + dev_precision_list[idx] + ' - format: ' + str(formats[idx]) + '\n')
    
    # Plot Recall
    sorted_idx = np.argsort(dev_recall_list)
    plot_data = []
    labels = []
    for idx in sorted_idx[:10]:
        plot_data.append(recall_list[idx])
        labels.append(str(formats[idx]))
    plot_images(plot_data, labels, "Epoch", "Recall", "Recall_VaryingActivationFuns.png")
    with open('Recall_VaryingActivationFuns.txt', 'w') as f:
        for idx in sorted_idx:
            f.write('dev_lost: ' + dev_recall_list[idx] + ' - format: ' + str(formats[idx]) + '\n')

def experiment_optimizers():
    optimizer_list = ['adagrad', 'sgd', 'rmsprop', 'adam']
    for loss_function in ['RMSE', 'precision', 'recall']: 
        epoch_loss_list = []
        dev_loss_list = []
        # define ACTIVATION_FORMAT
        for optimizer in optimizer_list:
            OPTIMIZER = optimizer
            print("Running on " + OPTIMIZER + " optimizer")
            (epoch_loss, dev_loss, test_loss) = train(LEARNING_RATE, WEIGHT_DECAY, loss_function, NUM_ITERATIONS, False)
            epoch_loss_list.append(epoch_loss)
            dev_loss_list.append(dev_loss)
        plot_images(epoch_loss_list, optimizer_list, "Epoch", loss_function, loss_function + "_VaryingOptimizers.png")
        with open(loss_function + '_VaryingOptimizers.txt', 'w') as f:
            for idx in range(optimizer_list):
                f.write('dev_lost: ' + dev_loss_list[idx] + ' - optimizer: ' + optimizer_list[idx] + '\n')

def experiment_hidden_layer():
    hidden_dimension_list = [5, 10, 20, 35, 50, 75]
    for loss_function in ['RMSE', 'precision', 'recall']: 
        epoch_loss_list = []
        dev_loss_list = []
        # define activation_format and optimizer
        for hidden_dimension in hidden_dimension_list:
            HIDDEN_DIMENSION = hidden_dimension
            print("Running on " + HIDDEN_DIMENSION + " hidden dimensions")
            (epoch_loss, dev_loss, test_loss) = train(LEARNING_RATE, WEIGHT_DECAY, loss_function, NUM_ITERATIONS, False)
            epoch_loss_list.append(epoch_loss)
            dev_loss_list.append(dev_loss)
        plot_images(epoch_loss_list, optimizer_list, "Epoch", loss_function, loss_function + "_VaryingHiddenDim.png")
        plot_dev_loss(hidden_dimension_list, dev_loss_list, "Number of Hidden Dimensions", loss_function, loss_function + "_VaryingHiddenDim_Dev.png")

def experiment_stack_number():
    return 0

def experiment_learning_rate():
    learning_rates = [i / 100.0 for i in range(1, 10)]
    for loss_function in ['RMSE', 'precision', 'recall']: 
        # define activation_format, optimizer, and hiddem dimensions
        plot_data = []
        labels = []
        dev_loss_list = []
        for learning_rate in learning_rates:
            (epoch_loss, dev_loss, test_loss) = train(LEARNING_RATE, WEIGHT_DECAY, "MMSE", NUM_ITERATIONS, True)
            plot_data.append(epoch_loss)
            labels.append("Learning rate: " + str(learning_rate))
            dev_loss_list.append(dev_loss)
        plot_images(plot_data, labels, "Epoch", loss_function, loss_function + "_VaryingLearningRate.png")
        plot_dev_loss(learning_rates, dev_loss_list, "Learning rate", loss_function, loss_function + "_VaryingLearningRateDev.png")

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
    experiment_activation_function()
#    experiment_optimizers()
#    experiment_hidden_layer()
#    experiment_stack_number()
#    experiment_learning_rate()
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
