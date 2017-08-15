#!/usr/bin/env python
""" run_plots.py:
Testing plotting functions using ini files to load data.
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
config.read('compare_dqn_imit.ini')

plot = config['DEFAULT']
data = config['data']
x = config['x_axis']
y = config['y_axis']

# plot setup
plt.figure()
sns.set()
plt.xlabel(x['label'], fontsize=plot['fontsize'])
plt.ylabel(y['label'], fontsize=plot['fontsize'])
plt.suptitle(plot['title'], fontsize=plot['fontsize'])

# loop to load and plot data
for i in range(data.getint('n_data')):
    # load it
    if plot['mode'] == 'compare':
        plot_data = np.genfromtxt(data['address'+str(i)], delimiter=',')

        # plot
        plt.plot(plot_data[:,0],
                 plot_data[:,1:],
                 data['color'+str(i)],
                 label=data['label'+str(i)])

    elif plot['mode'] == 'avg':
        steps, rew = compute_avg(data, i)

        # plot avg and std dev
        sns.tsplot(time=steps, data=rew,
                   color=data['color'+str(i)],
                   linestyle='-',
                   condition=data['label'+str(i)])

# axis limits and legend
plt.xlim([x.getfloat('low'),x.getfloat('high')])
plt.legend(loc='lower right',fontsize=plot['fontsize'])

# show and save
if args.save:
    plt.savefig(plot['save_dir']+plot['save_name']+'.png')
if args.show:
    plt.show()
print('Done!')
