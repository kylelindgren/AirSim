#!/usr/bin/env python
""" validate_depth.py
Translate distance to pixel value of depth sensor.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "July 10, 2017"

# import
import numpy as np
import cv2
import time
from scipy import misc
import matplotlib.pyplot as plt
from matplotlib import pylab
import seaborn as sns

def main():
    """
    Load file with distance and pixel location, check pixel values.
    """
    # load distances and image
    depth_vals = np.genfromtxt('../figures/validate_depth/depth_vals', delimiter=',')
    img = misc.imread('../figures/validate_depth/depth.png')

    # get pixel values
    samples = depth_vals.shape[0]
    dist = np.zeros(samples)
    pixel = np.zeros(samples)

    for i in range(samples):
        dist[i] = depth_vals[i][0]
        pixel[i] = img[int(depth_vals[i][1]), int(depth_vals[i][2])]

    # fit data with polynomial
    z = np.polyfit(dist, pixel, 3)
    f = np.poly1d(z)

    dist_new = np.linspace(dist[0],dist[-1],50)
    pixel_new = f(dist_new)

    # plot for quick visualization
    plt.figure()
    sns.set()
    plt.plot(dist, pixel,'o',label='data')
    plt.plot(dist_new, pixel_new, '--',label='fit')
    plt.title('Pixel Value Variation with Distance', fontsize='medium')
    plt.xlabel('Distance [m]', fontsize='medium')
    plt.ylabel('Pixel [unit]', fontsize='medium')
    plt.ylim([90,260])
    plt.xlim([0,20])
    plt.legend(loc='best', fontsize='medium')
    plt.savefig('../figures/validate_depth/pixel_distance.png')

if __name__ == '__main__':
    main()
