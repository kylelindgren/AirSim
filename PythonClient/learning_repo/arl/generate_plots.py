#!/usr/bin/env python
""" generate_plots.py:
Load data and generate reports.
Testing the use of SEABORN library.

Wanted to have reports off:
- Training results (loss and acc): 2 plots
- Total reward per episode: 1 plot
- Avg and stddev reward per step: 1 plot

Can add different methods on the same plots for comparison purposes.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "June 16, 2017"

# import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import argparse


class GeneratePlots():
    """
    Class to generate all plots required for the report.
    """

    def __init__(
            self,
            data_folder,
            fig_folder,
            train_legend,
            run_id,
            train_name,
            n_episodes,
            n_epochs,
            save_figs):
        self.data_folder = data_folder
        self.fig_folder = fig_folder
        self.run_id = run_id
        self.train_legend = train_legend
        self.train_name = train_name
        self.n_episodes = n_episodes
        self.n_epochs = n_epochs
        self.save_figs = save_figs
        self.c = ["#4878CF", "#6ACC65", "#D65F5F",
                  "#B47CC7", "#C4AD66", "#77BEDB"]  # base color

    def plot_train_hist(self, hist, data_type):
        """
        Plot training history.
        """
        # plot setup
        plt.figure()
        plt.xlabel('Epoch [unit]', fontsize='medium')

        if data_type == 'loss':
            plt.suptitle('Training Loss per Epoch', fontsize='medium')
            plt.ylabel('Mean Squared Error [unit]', fontsize='medium')
            cut_factor = .02  # in %
            plt.ylim([0, .3])

        elif data_type == 'acc':
            plt.suptitle('Training Accuracy per Epoch', fontsize='medium')
            plt.ylabel('Accuracy [unit]', fontsize='medium')
            cut_factor = .0  # in %
            plt.ylim([0, 1])

        elif data_type == 'val_loss':
            plt.suptitle('Validation Loss per Epoch', fontsize='medium')
            plt.ylabel('Mean Squared Error [unit]', fontsize='medium')
            cut_factor = .02  # in %
            plt.ylim([0.2, .45])

        elif data_type == 'val_acc':
            plt.suptitle('Validation Accuracy per Epoch', fontsize='medium')
            plt.ylabel('Accuracy [unit]', fontsize='medium')
            cut_factor = .0  # in %
            plt.ylim([0, 1])

        # remove beginning of loss (too high) and fix axis
        # cut first x% of the history (for scaling purposes)
        hist_init = int(cut_factor * hist.shape[0])

        # plot with seaborn
        for i in range(len(self.train_name)):
            sns.set()
            plt.plot(hist[hist_init:, 0], hist[hist_init:, 1 + i],
                     self.c[i], label=self.train_legend[i])

        # axis limits
        plt.xlim([0, self.n_epochs])
        plt.legend(loc='best')

        if self.save_figs:
            plt.savefig(
                self.fig_folder +
                self.run_id +
                '_' +
                data_type +
                '.png')

    def plot_avg_rew(self, steps, rew):
        """
        Load files, calculate average, std dev, plot, and save figure.
        """
        # plto setup
        plt.figure()
        plt.xlabel("Time Step [unit]", fontsize='medium')
        plt.suptitle("Average Reward per Time Step", fontsize='medium')
        plt.ylabel("Reward [unit]", fontsize='medium')
        # plt.ylim([0.4, .8])

        # plot avg and std dev
        sns.tsplot(time=steps, data=rew, color=self.c, linestyle='-')
        if self.save_figs:
            plt.savefig(self.fig_folder + self.run_id + '_avg_rew.png')

    def plot_total_rew(self, steps, total_rew):
        """
        Load files, calculate average, std dev, plot, and save figure.
        """
        # plto setup
        plt.figure()
        plt.xlabel("Episode [unit]", fontsize='medium')
        plt.suptitle("Total Reward per Episode", fontsize='medium')
        plt.ylabel("Reward [unit]", fontsize='medium')

        # plot avg and std dev
        steps = steps + 1  # add 1 because first episode is the number 1
        sns.tsplot(time=steps, data=total_rew, color=self.c, linestyle='-')
        plt.xlim([0, self.n_episodes])
        if self.save_figs:
            plt.savefig(self.fig_folder + self.run_id + '_total_rew.png')

    def run(self, show_figs=True):
        # ----------------
        # Training plots
        # ----------------
        # load data
        data_types = ['loss', 'acc', 'val_loss', 'val_acc']

        for data_type in data_types:
            history = np.zeros((self.n_epochs, 1))

            for each_name in self.train_name:
                data_address = self.data_folder + self.run_id + each_name + data_type + '.csv'
                print("Loading ", data_address)
                temp_history = np.genfromtxt(data_address, delimiter=',')
                history = np.hstack(
                    (history, temp_history[:, 1].reshape(self.n_epochs, 1)))

            # plot
            history[:, 0] = temp_history[:, 0]
            self.plot_train_hist(history, data_type)

        # ----------------
        # Average reward plot
        # ----------------
        # load files
        root_file = self.data_folder + self.run_id + '_avg_'
        rew = np.genfromtxt(root_file + '0.csv', delimiter=',')

        # discarding actions and limiting step numbers to first file
        # time steps, rew, actions
        max_steps = 575 # MANUAL FOR NOW (CHECK THE LOWEST STEP NUMBER BEFORE)
        steps = rew[:max_steps, 0]
        rew = rew[:max_steps, 1].reshape(1, max_steps)

        # append other files to compute stddev and avg
        for i in range(self.n_episodes):
            # load additional data
            temp_rew = np.genfromtxt(
                root_file + str(i) + '.csv', delimiter=',')
            # print(len(temp_rew)) # THIS IS TO CHECK THE MIN NUMBER OF STEPS
                                   # COMMENT THE TWO BOTTOM LINES

            # discarding actions and limiting step numbers to first file
            temp_rew = temp_rew[:max_steps, 1].reshape(1, max_steps)

            # apprend rewards only to original dataset
            rew = np.append(rew, temp_rew, axis=0)

        # plot
        self.plot_avg_rew(steps, rew)

        # ----------------
        # Total reward plot
        # ----------------
        # load files
        root_file = self.data_folder + self.run_id + '_rew'
        total_rew = np.genfromtxt(root_file + '.csv', delimiter=',')

        self.plot_total_rew(total_rew[:, 0], total_rew[:, 1])

        # ----------------
        # show plots and save pics
        # ----------------
        print('Done!')
        if show_figs:
            plt.show()


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", help="show plots")
    parser.add_argument("--save", help="show plots")
    args = parser.parse_args()

    # ----------------
    # General parameters
    # ----------------
    data_folder = '../data/'
    fig_folder = '../figures/'
    run_id = 'exp2'
    n_episodes = 20  # number of run for reward plot
    n_epochs = 150
    save_figs = False
    show_figs = False
    if args.show:
        show_figs = True
    if args.save:
        save_figs = True

    # ----------------
    # Training results
    # ----------------
    train_name = ['_plot_hist_', '_plot_hist_big']
    train_legend = ['Decreasing Units', 'Increasing Units']

    # ----------------
    # Create report
    # ----------------
    report = GeneratePlots(
        data_folder,
        fig_folder,
        train_legend,
        run_id,
        train_name,
        n_episodes,
        n_epochs,
        save_figs)
    report.run(show_figs)
