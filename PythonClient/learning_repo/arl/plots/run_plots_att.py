#!/usr/bin/env python
""" run_plots_att.py:
Testing plotting functions using ini files to load data to compare attitude.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "June 18, 2017"

# import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import configparser

# define some support functions
def compute_avg(data, idx):
    """
    Compute average of data sequence.
    """
    # load files
    rew = np.genfromtxt(data['address'+str(idx)] + '0.csv', delimiter=',')

    # discarding actions and limiting step numbers to first file
    # time steps, rew, actions
    max_steps = data.getint('n_steps')
    steps = rew[:max_steps, 0]
    rew = rew[:max_steps, 1].reshape(1, max_steps)

    # append other files to compute stddev and avg
    for i in range(1,data.getint('n_episodes')):
        # load additional data
        temp_rew = np.genfromtxt(
            data['address'+str(idx)] + str(i) + '.csv', delimiter=',')

        # discarding actions and limiting step numbers to first file
        temp_rew = temp_rew[:max_steps, 1].reshape(1, max_steps)

        # apprend rewards only to original dataset
        rew = np.append(rew, temp_rew, axis=0)


    return steps, rew


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--show", help="show plots")
parser.add_argument("--save", help="show plots")
args = parser.parse_args()

# start config parser
config = configparser.ConfigParser()
config.read('test_att.ini')

plot = config['DEFAULT']

# parse for each desired plot
for j in range(plot.getint('n_plots')):
    data = config['data'+str(j)]

    # plot setup
    plt.figure()
    plt.xlabel(data['x_label'], fontsize=plot['fontsize'])
    plt.ylabel(data['y_label'], fontsize=plot['fontsize'])
    plt.suptitle(data['title'], fontsize=plot['fontsize'])

    # loop to load and plot data
    for i in range(data.getint('n_data')):
        # load it
        if data['mode'] == 'compare':
            plot_data = np.genfromtxt(data['address'+str(i)], delimiter=',')

            # plot
            sns.set()
            plt.plot(plot_data[:,0],
                     plot_data[:,data.getint('col'+str(i))],
                     plot['color'+str(i)],
                     ls=data['ls'+str(i)],
                     label=data['label'+str(i)])

        elif data['mode'] == 'avg':
            steps, rew = compute_avg(data, i)

            # plot avg and std dev
            sns.tsplot(time=steps, data=rew,
                       color=plot['color'+str(i)],
                       linestyle='-',
                       condition=data['label'+str(i)])

    # axis limits and legend
    if plot.getboolean('tight_layout'):
        plt.tight_layout()
    plt.xlim([data.getfloat('x_low'),data.getfloat('x_high')])
    plt.ylim([data.getfloat('y_low'),data.getfloat('y_high')])
    plt.legend(loc='best',fontsize=plot['fontsize'])

    # show and save
    if args.save:
        plt.savefig(plot['save_dir']+data['save_name']+'.png')
    if args.show:
        plt.show()

print('Done!')
