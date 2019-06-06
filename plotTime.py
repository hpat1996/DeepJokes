import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.ioff()

def plot_images(plot_data, labels, xlabel, ylabel, filename):
    refined_data = []
    for data in plot_data:
        refined_data.append(list(filter(lambda x: x[0] < 100 and x[1] < 5, data)))

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


def timeError(lines):
    lines   = [line.replace('\n', '') for line in lines]
    times   = [float(time) for time in lines[0].split(',')]
    errors  = [float(error) for error in lines[1].split(',')]

    return list(zip(times, errors))

with open("images/Time_Error.txt", "r") as f:
    lines   = f.readlines()
    deep    = lines[0:4]
    stacked = lines[4:8]
    latent  = lines[8:12]

    plot_data_train = []
    plot_data_train.append(timeError(deep[0:2]))
    plot_data_train.append(timeError(stacked[0:2]))
    plot_data_train.append(timeError(latent[0:2]))

    plot_data_dev   = []
    plot_data_dev.append(timeError(deep[2:4]))
    plot_data_dev.append(timeError(stacked[2:4]))
    plot_data_dev.append(timeError(latent[2:4]))
 
    labels  = ['StackedAutoencoder', 'DeepAutoencoder', 'LatentFeatures']
    plot_images(plot_data_train,    labels, "Time (seconds)", "Root Mean Squared Error", "images/Time_Error_Train.png")
    plot_images(plot_data_dev,      labels, "Time (seconds)", "Root Mean Squared Error", "images/Time_Error_Dev.png")
