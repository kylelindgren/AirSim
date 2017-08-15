#!/usr/bin/env python
""" evaluate.py:
Evaluate performance of learning algorithms by training different networks and
comparing them.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "June 13, 2017"

# import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # remove TF warning about CPU
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, SGD, Adam
from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from neural import save_neural

tf.logging.set_verbosity(tf.logging.INFO)

def create_model_0(height, width):
    '''
    Create model to be evaluated.
    '''
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(height, width, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # the model so far outputs 3D feature maps (height, width, features)

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def compare_imitation():
    '''
    Create imitation networks for different amounts of data. Later evaluate these
    different networks.
    '''
    # parameters
    height = 36
    width = 64
    n_act = 1
    run_id = 'exp2'
    n_items_init = 1
    n_items = 3 # number of runs. it will be created one net per run

    # loop training a model for each amount of runs desired
    for i in range(n_items_init, n_items):
        print('******** Run {} out of {} ********'.format(i+1,n_items))
        # create model
        # can initialize using different functions to test different architectures
        model = create_model_0(height, width)

        # load files
        print('Loading files ...')
        root_file = '../data/'+run_id+'_imit_'
        name_file = root_file+'0.csv'
        print('Loading ', name_file)
        data = np.genfromtxt(name_file, delimiter=',')

        for j in range (i):
            # load each other file
            name_file = root_file + str(j+1) +'.csv'
            print('Loading ', name_file)
            temp_data = np.genfromtxt(name_file, delimiter=',')

            # apprend temp_data to original dataset
            data = np.vstack((data,temp_data))

        # split states and actions
        x = data[:,:-n_act] / 255
        y = data[:,-n_act:]

        # capture oroiginal size of features and samples
        samples = x.shape[0]
        features = x.shape[1]
        print('Number of samples: ', samples)
        print('x features: ', features)
        print('y features: ', y.shape[1])

        # reshape to sequence
        # samples, height, width, channels
        x = x.reshape((samples, height, width, 1))
        y = y.reshape((samples, n_act))

        # train
        model.fit(x, y, batch_size=32,
                         epochs=100,
                         shuffle=False,
                         verbose=1)

        # plot train history
        save_neural(model, 'eval_imit_'+str(i))

def organize_rew():
    '''
    Load csv files with reward results and organize in columns so it can be
    plotted using seaborn.
    '''
    # parameters
    n_files = 20
    n_runs = 5

    # create var to save rewards
    plot_rew = np.zeros((n_files+1,n_runs))
    plot_human = np.zeros((n_files+1,n_runs))

    # load initial file
    root_file = '../data/eval_imit_'
    name_file = root_file+'0_rew.csv'
    print('Loading ', name_file)
    data = np.genfromtxt(name_file, delimiter=',')
    plot_rew[1,:] = data[:,1]

    for i in range (1,n_files):
        # load each other file
        name_file = root_file + str(i) +'_rew.csv'
        print('Loading ', name_file)
        temp_data = np.genfromtxt(name_file, delimiter=',')

        # apprend temp_data to original dataset
        plot_rew[i+1,:] = temp_data[:,1]

    # load human baseline
    name_file = '../data/human_eval_imit_rew.csv'
    print('Loading human baseline: ', name_file)
    data = np.genfromtxt(name_file, delimiter=',')
    plot_human[:,:] = data[:,1]

    # plot
    # base colors: ["#4878CF", "#6ACC65", "#D65F5F","#B47CC7", "#C4AD66", "#77BEDB"]

    plt.figure()
    sns.set()

    plt.xlabel('Demonstrations [unit]',fontsize='medium')
    plt.ylabel('Total Reward [unit]',fontsize='medium')
    plt.suptitle('Learning Agent Performance per Demonstrations',fontsize='medium')

    steps = np.arange(n_files+1) # +1 for the zero (untrained)
    # machine
    sns.tsplot(time=steps, data=plot_rew.T,
               color='#4878CF',
               linestyle='-',
               condition='Learning Agent')
    # human
    sns.tsplot(time=steps, data=plot_human.T,
               color='#6ACC65',
               linestyle='-',
               condition='Human Baseline')

    # axis limits and legend
    plt.xlim([0,20])
    plt.ylim([350,500])
    plt.legend(loc='best',fontsize='medium')

    # plt.show()
    plt.savefig('../figures/evaluate_imit.png', transparent = False)


if __name__ == '__main__':
    # train all networks
    # compare_imitation()

    # organize reward files
    organize_rew()
