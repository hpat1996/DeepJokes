import numpy as np
import pandas as pd
import torch
import random
import math
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

f = open("../data/jester-data-1.csv")
users = 24983
jokes = 100

def mapping(i):
    if i == 99:
        return 99
    return math.ceil((i+10.0)/4.0)

training_set = np.full((users, jokes), 99)
k = 0
for line in f:
    entry = line.split(",")
    vals = np.arange(1, jokes)
    np.random.shuffle(vals)
    for i in range(0, len(vals) - 30):
        training_set[k][vals[i]-1] = mapping(float(entry[vals[i]][1:-1]))
    k += 1
print training_set[0]
print training_set.shape
iterations = 100

class StackedAutoEncoder(nn.Module):
    def __init__(self, ):
        super(StackedAutoEncoder, self).__init__()
        self.ae1 = nn.Sequential(
            nn.Linear(jokes, 5),
            nn.Sigmoid(),
            nn.Linear(5, jokes)
        )
        self.ae2 = nn.Sequential(
            nn.Linear(jokes, 5),
            nn.Sigmoid(),
            nn.Linear(5, jokes)
        )
        self.ae3 = nn.Sequential(
            nn.Linear(jokes, 5),
            nn.Sigmoid(),
            nn.Linear(5, jokes)
        )
        self.ae4 = nn.Sequential(
            nn.Linear(jokes, 5),
            nn.Sigmoid(),
            nn.Linear(5, jokes)
        )

    def forward(self, x):
        x = self.ae1(x)
        x = self.ae2(x)
        x = self.ae3(x)
        x = self.ae4(x)
        return x

stackedAutoEncoder = StackedAutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(stackedAutoEncoder.parameters(), lr = 0.000001, weight_decay=0.8)
training_set = torch.FloatTensor(training_set)
def train():
    for i in range(0, iterations):
        train_loss = 0
        s = 0.
        for id_user in range(users):
            input = Variable(training_set[id_user]).unsqueeze(0)
            target = input.clone()
            output = stackedAutoEncoder(input)
            target.require_grad = False
            output[target == 99] = 99
            loss = criterion(output, target)
            mean_corrector = jokes/float(torch.sum(target.data != 99) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()
        print('epoch: '+str(i)+' loss: '+str(train_loss/s))
train()
