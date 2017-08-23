#!/usr/bin/env python
""" test_ini.py:
Testing ini files to automate experiments.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "June 18, 2017"

# import
import configparser

config = configparser.ConfigParser()
print(config.sections())
print(config.read('example.ini'))
print(config.sections())

print('bitbucket.org' in config)
print('bitbuckets.org' in config)

print(config['DEFAULT']['Compression'])

for key in config['bitbucket.org']: print(key)

####
units = config['neural_net'].getint('Units')
name = config['neural_net']['Name']
print(units)
print(units.__class__)
print(name.__class__)

nn = config['neural_net']
print(nn.get('Name'))
print(nn['Name'])

ts = config['topsecret.server.com']
print(ts.get('Secret'))
