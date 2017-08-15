#!/usr/bin/env python
""" plot.py:
Plotting tools for ARL project.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "June 14, 2017"

# import
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns


def plot_history(history, std_dev=None, reduced=True, save_pic=False, name_pic='hist.png'):
    '''
    Plot history results (iterations, mean reward) using Matplotlib.
    '''
    # extract parameters from data
    steps = history[:, 0]
    rew = history[:, 1]

    # plot rewards
    sns.set()
    fig, ax = plt.subplots(1)
    ax.set_title("Training Rewards", fontsize='medium')
    ax.set_xlabel("Episode [unit]", fontsize='medium')
    ax.set_ylabel("Reward [unit]", fontsize='medium')
    ax.plot(steps, rew, '-k', alpha=.75)

    # check for std dev
    if std_dev is not None:
        # print std dev areas
        ax.fill_between(steps, rew + std_dev, rew - std_dev, facecolor='green', alpha=0.35)

    # save/show pic
    if save_pic:
        plt.savefig(name_pic)
    else:
        plt.show()

def process_avg(run_id, n_items=3):
    """
    Load files, calculate average, std dev, plot, and save figure.
    """
    # load files
    root_file = '../data/'+run_id+'_avg_'
    data = np.genfromtxt(root_file+'0.csv', delimiter=',')

    for i in range (1,n_items):
        # load each other file
        temp_data = np.genfromtxt(root_file + str(i) +'.csv', delimiter=',')
        rewards = temp_data[:,1].reshape(len(data),1)

        # apprend rewards only to original dataset
        data = np.hstack((data,rewards))

    # average data
    avg = np.average(data[:,1:], axis=1).reshape(len(data),1)
    std_dev = np.std(data[:,1:], axis=1)#.reshape(len(data),1)

    avg_plot = np.hstack((data[:,0].reshape(len(data),1), avg))

    # plot
    plot_history(avg_plot, std_dev, save_pic=False, name_pic=run_id+'.png')

if __name__ == '__main__':
    process_avg('exp1',20)
