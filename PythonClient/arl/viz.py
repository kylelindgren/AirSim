#!/usr/bin/env python
""" viz.py:
Visualization techniques for the neural network and tools to help compare and
evaluate the behavior learned.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "June 26, 2017"

# import
import numpy as np
import h5py

from neural import load_neural, ImitationNetwork, save_neural

import matplotlib.pyplot as plt
import seaborn as sns

def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param))
    finally:
        f.close()

def get_layer_info(model):
    """
    Print info of non-zeros layers of a model.
    """
    # array to save norms
    norms = []
    # extract layer info
    n_layers = len(model.layers)
    for i in range(n_layers):
        layer = model.get_layer(index=i)
        n_weights = len(layer.get_weights())

        # check if have weights
        if n_weights != 0:
            print("# Layer class: ", layer.__class__)
            print("  # of Weight Arrays: ", n_weights)
            for j in range(n_weights):
                weight_array =  np.asarray(layer.get_weights()[j]).flatten()
                print("    # Sublayer %i:" %j)
                print("      # of Weights: %i" %weight_array.shape[0])
                calc_norm = np.linalg.norm(weight_array)
                print("      # Norm: %.2f" %calc_norm)
                norms.append(calc_norm)

    return norms

def count_nonzero_layers(model):
    """
    Return number of nonzero layers of a model.
    """
    count = 0
    # extract layer info
    n_layers = len(model.layers)
    for i in range(n_layers):
        # get weights and check if zero
        layer = model.get_layer(index=i)
        n_weights = len(layer.get_weights())

        # check if have weights
        if n_weights != 0:
            count +=1

    return count

def compare_norms():
    """
    Load network, compara the norms of their weights.
    """
    names = ['trained_imit_exp2_big', 'trained_imit_exp2_']
    colors = ['r', 'g']
    legends = ['big', 'small']
    for j in range(len(names)):
        # load network
        model = load_neural(name=names[j], loss='mse', opt='adam')

        print('\n*** NAME: %s ***' %names[j])
        print('* Layers with weights: ', count_nonzero_layers(model))

        # print info
        norms = get_layer_info(model)
        sns.tsplot(norms, color=colors[j], condition=legends[j])

def plot_train_hist_compare(history1, history2, history3, history4):
    """
    Plot training history.
    """
    data_folder = '../data/'
    plot_items = ['loss','acc','val_loss','val_acc']

    for item in plot_items:
        plt.figure()
        plt.xlabel('Epoch [unit]', fontsize='medium')
        if item == 'loss':
            plt.suptitle('Training Loss', fontsize='medium')
            plt.ylabel('Mean Squared Error [unit]', fontsize='medium')
            cut_factor = .0 # in %
        elif item == 'acc':
            plt.suptitle('Training Accuracy', fontsize='medium')
            plt.ylabel('Accuracy [unit]', fontsize='medium')
            cut_factor = .0 # in %
        elif item == 'val_loss':
            plt.suptitle('Validation Loss', fontsize='medium')
            plt.ylabel('Mean Squared Error [unit]', fontsize='medium')
            cut_factor = .0 # in %
        elif item == 'val_acc':
            plt.suptitle('Validation Accuracy', fontsize='medium')
            plt.ylabel('Accuracy [unit]', fontsize='medium')
            cut_factor = .0 # in %

        # (hist 1) convert to numpy array and reshape as a column to save data
        hist_list1 = np.asarray(history1.history[item])
        hist_list1 = hist_list1.reshape(len(hist_list1),1)
        steps1 = np.arange(len(hist_list1)).reshape(len(hist_list1),1)

        # (hist 2) convert to numpy array and reshape as a column to save data
        hist_list2 = np.asarray(history2.history[item])
        hist_list2 = hist_list2.reshape(len(hist_list2),1)
        steps2 = np.arange(len(hist_list2)).reshape(len(hist_list2),1)

        # (hist 3) convert to numpy array and reshape as a column to save data
        hist_list3 = np.asarray(history3.history[item])
        hist_list3 = hist_list3.reshape(len(hist_list3),1)
        steps3 = np.arange(len(hist_list3)).reshape(len(hist_list3),1)

        # (hist 4) convert to numpy array and reshape as a column to save data
        hist_list4 = np.asarray(history4.history[item])
        hist_list4 = hist_list4.reshape(len(hist_list4),1)
        steps4 = np.arange(len(hist_list4)).reshape(len(hist_list4),1)

        # save plot data
        save_plot1 = np.hstack((steps1,hist_list1))
        save_plot2 = np.hstack((steps2,hist_list2))
        save_plot3 = np.hstack((steps3,hist_list3))
        save_plot4 = np.hstack((steps4,hist_list4))

        np.savetxt(data_folder+'batch_hist_'+item+'.csv', save_plot1, delimiter=',')
        np.savetxt(data_folder+'single_hist_'+item+'.csv', save_plot2, delimiter=',')
        np.savetxt(data_folder+'10_hist_'+item+'.csv', save_plot3, delimiter=',')
        np.savetxt(data_folder+'20_'+item+'.csv', save_plot4, delimiter=',')

        # plot
        plt.plot(steps1, hist_list1, label='32 samples')
        plt.plot(steps2, hist_list2, label='1 sample')
        plt.plot(steps3, hist_list3, label='10 samples')
        plt.plot(steps4, hist_list4, label='20 samples')
        plt.legend()

        plt.savefig('../figures/compare_'+item+'.png')

def compare_batchs():
    """
    Load network, then train it using big and small minibatchs. Compare the
    "Euclidean distance" between final network and the original one. They should
    have similar distances by the end of training.
    """
    # load networks
    print('Loading neural network...')
    model_batch = load_neural(name='trained_imit_exp2_', loss='mse', opt='adam')
    model_single = load_neural(name='trained_imit_exp2_', loss='mse', opt='adam')
    model_10 = load_neural(name='trained_imit_exp2_', loss='mse', opt='adam')
    model_20 = load_neural(name='trained_imit_exp2_', loss='mse', opt='adam')

    # load new data to be trained
    data = np.genfromtxt('../data/exp2_imit_10.csv', delimiter=',')

    # split states and actions
    x = data[:,:-1]
    y = data[:,-1]

    # reshape arrays to 2d images
    # samples, height, width, channels
    samples = x.shape[0]
    print('Number of samples: ', samples)
    print('x features: ', x.shape[1])

    # depth sensor: 36 x 64
    x = x.reshape((samples, 36, 64, 1))
    y = y.reshape((samples, 1))

    print('y features: ', y.shape[1])

    # train
    n_epochs = 150
    hist_train_batch = model_batch.fit(x, y, batch_size=32,
                     epochs=n_epochs,
                     validation_split=.2,
                     verbose=2)

    hist_train_single = model_single.fit(x, y, batch_size=1,
                     epochs=n_epochs,
                     validation_split=.2,
                     verbose=2)

    hist_train_10 = model_10.fit(x, y, batch_size=10,
                     epochs=n_epochs,
                     validation_split=.2,
                     verbose=2)

    hist_train_20 = model_20.fit(x, y, batch_size=20,
                     epochs=n_epochs,
                     validation_split=.2,
                     verbose=2)


    # plot train history
    # save_neural(self.model, 'trained_imit_exp2_batch')
    # save_neural(self.model, 'trained_imit_exp2_single')
    plot_train_hist_compare(hist_train_batch, hist_train_single, hist_train_10, hist_train_20)

def compare_batchs2():
    """
    Load 50 percent of the data, and compare its performance based on training
    the network on the rest of the data using different batch values.
    """
    # create network
    net = ImitationNetwork()
    model = net.model

    # load files
    n_epochs = 50
    n_items = 10 # 10 first runs
    run_id = 'exp2'
    print('Loading files ...')
    root_file = '../data/'+run_id+'_imit_'
    data_folder = '../data/'
    data = np.genfromtxt(root_file+'0.csv', delimiter=',')

    for i in range (1,n_items):
        # load each other file
        temp_data = np.genfromtxt(root_file + str(i) +'.csv', delimiter=',')

        # apprend temp_data to original dataset
        data = np.vstack((data,temp_data))

    # split states and actions
    x = data[:,:-1]
    y = data[:,-1]

    # reshape arrays to 2d images
    # samples, height, width, channels
    samples = x.shape[0]
    print('Number of samples: ', samples)
    print('x features: ', x.shape[1])

    x = x.reshape((samples, net.height, net.width, 1))
    y = y.reshape((samples, 1))

    print('y features: ', y.shape[1])

    # train
    hist_train = model.fit(x, y, batch_size=32,
                     epochs=n_epochs,
                     validation_split=.2,
                     verbose=2,
                     shuffle=True)

    # plot train history
    save_neural(net.model, 'original_compare_small')
    plot_items = ['loss','acc','val_loss','val_acc']

    for item in plot_items:
        # convert to numpy array and reshape as a column to save data
        hist_list = np.asarray(hist_train.history[item])
        hist_list = hist_list.reshape(len(hist_list),1)
        steps = np.arange(len(hist_list)).reshape(len(hist_list),1)

        # save plot data
        save_plot = np.hstack((steps,hist_list))
        np.savetxt(data_folder+'compare_hist_original_small_'+item+'.csv', save_plot, delimiter=',')


    # load rest of data
    # load files
    n_items = 20
    print('Loading files ...')
    data = np.genfromtxt(root_file+'11.csv', delimiter=',')

    for i in range (12,n_items):
        # load each other file
        temp_data = np.genfromtxt(root_file + str(i) +'.csv', delimiter=',')

        # apprend temp_data to original dataset
        data = np.vstack((data,temp_data))

    # split states and actions
    x = data[:,:-1]
    y = data[:,-1]

    # reshape arrays to 2d images
    # samples, height, width, channels
    samples = x.shape[0]
    print('Number of samples: ', samples)
    print('x features: ', x.shape[1])

    x = x.reshape((samples, net.height, net.width, 1))
    y = y.reshape((samples, 1))

    print('y features: ', y.shape[1])

    # train different network using different values
    data_folder = '../data/'
    plot_items = ['loss','acc','val_loss','val_acc']
    net_id = ['1','5','10','20','32']

    for j in range(len(net_id)):

        # create and train network
        temp_net = ImitationNetwork()
        temp_net.model = net.model

        # train
        hist_train = temp_net.model.fit(x, y, batch_size=int(net_id[j]),
                         epochs=n_epochs,
                         validation_split=.2,
                         verbose=2,
                         shuffle=False)

        for item in plot_items:

            # convert to numpy array and reshape as a column to save data
            hist_list = np.asarray(hist_train.history[item])
            hist_list = hist_list.reshape(len(hist_list),1)
            steps = np.arange(len(hist_list)).reshape(len(hist_list),1)

            # save plot data
            save_plot = np.hstack((steps,hist_list))
            np.savetxt(data_folder+'compare_hist_net_small_'+str(net_id[j])+'_'+item+'.csv', save_plot, delimiter=',')

def compare_batchs3():
    """
    Load 50 percent of the data, and compare its performance based on training
    the network on the rest of the data using different batch values.
    """
    # create network
    net = ImitationNetwork()
    print('Loading neural network...')
    net.model = load_neural(name='original_compare', loss='mse', opt='adam')

    # load files
    n_epochs = 50
    n_items = 10 # 10 first runs
    run_id = 'exp2'
    print('Loading files ...')
    root_file = '../data/'+run_id+'_imit_'
    data_folder = '../data/'

    # load rest of data
    # load files
    n_items = 20
    print('Loading files ...')
    data = np.genfromtxt(root_file+'11.csv', delimiter=',')

    for i in range (12,n_items):
        # load each other file
        temp_data = np.genfromtxt(root_file + str(i) +'.csv', delimiter=',')

        # apprend temp_data to original dataset
        data = np.vstack((data,temp_data))

    # split states and actions
    x = data[:,:-1]
    y = data[:,-1]

    # reshape arrays to 2d images
    # samples, height, width, channels
    samples = x.shape[0]
    print('Number of samples: ', samples)
    print('x features: ', x.shape[1])

    x = x.reshape((samples, net.height, net.width, 1))
    y = y.reshape((samples, 1))

    print('y features: ', y.shape[1])

    # train different network using different values
    data_folder = '../data/'
    plot_items = ['loss','acc','val_loss','val_acc']
    net_id = ['1','5','10','20','32']

    for j in range(len(net_id)):

        # create and train network
        temp_net = ImitationNetwork()
        temp_net.model = net.model

        # train
        hist_train = temp_net.model.fit(x, y, batch_size=int(net_id[j]),
                         epochs=n_epochs,
                         validation_split=.2,
                         verbose=2,
                         shuffle=True)

        for item in plot_items:

            # convert to numpy array and reshape as a column to save data
            hist_list = np.asarray(hist_train.history[item])
            hist_list = hist_list.reshape(len(hist_list),1)
            steps = np.arange(len(hist_list)).reshape(len(hist_list),1)

            # save plot data
            save_plot = np.hstack((steps,hist_list))
            np.savetxt(data_folder+'compare_hist_net_sf_'+str(net_id[j])+'_'+item+'.csv', save_plot, delimiter=',')


if __name__ == '__main__':
    # weight_file_path = '../neural_models/trained_imit_exp2_big.h5'
    # model_file_path = '../neural_models/trained_imit_exp2_big.json'
    #
    # print_structure(weight_file_path)
    compare_batchs3()
