#!/usr/bin/env python
"""Test Anaconda Environment

Run this script after you installed the Anaconda environment.
It will run Python and import some basic modules just to make sure everyhting
was installed correctly.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "May 16, 2017"

# import
print('Importing modules...')

import PIL
import scipy
import h5py
import matplotlib
import sklearn
import cv2
import gym
#import universe
import keras
import numpy as np
import tensorflow as tf

print('All modules were imported correctly.\n')
print('Reporting versions:')

print('Keras: ', keras.__version__)
print('Numpy: ', np.__version__)
print('TensorFlow: ', tf.__version__)
#print('PIL: ', PIL.__version__)
print('Scipy: ', scipy.__version__)
print('H5py: ', h5py.__version__)
print('Matplotlib: ', matplotlib.__version__)
print('SKLearn: ', sklearn.__version__)
print('OpenCV: ', cv2.__version__)
print('OpenAI Gym: ', gym.__version__)
