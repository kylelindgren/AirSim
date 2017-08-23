#!/bin/bash

echo "ARL Anaconda Environment."
echo "Installing system dependencies..."
echo "You will probably be asked for your sudo password."
sudo apt-get update
sudo apt-get install -y python-pip python-dev swig cmake build-essential
sudo apt-get install -y g++ libblas-dev
sudo apt-get build-dep -y python-scipy
sudo apt-get install -y python-numpy zlib1g-dev libjpeg-dev \
xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev

# requirements for universe
sudo apt-get install -y golang libjpeg-turbo8-dev make

pip install --upgrade pip

echo "Creating conda environment..."
conda env create -f environment.yml
conda env update

echo "Conda environment created!"
