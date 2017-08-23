# Human-Machine Interaction - Autonomous Unmanned Air Vehicles
## Project Overview

It is proposed to use novel Deep Reinforcement Learning algorithms coupled with direct human input to
accelerate the training of a quadrotor to autonomously fly and navigate through an known
or unknown map and achieve a desired position or state, or execute a given task. This
implicitly requires maintain stable flight along a trajectory and perform collision avoidance
when necessary.  

The proposed approach aims to combine human-input to accelerate the policy learned
by the intelligent agent and maximize the goals of a given task, at the same time being
robust to changes in the environment.  

This project is being developed during internship at US Army Research Laboratory (ARL) - Adelphi, MD.  
May 16 to August 25, 2017.  

by Vinicius Guimaraes Goecks  

## Current Results

The video below shows the current result of the project. The drone is able to navigate through a dense forest and avoid imminent obstacles. The policy was trained using imitation learning, based in approximately 40 min of human data performing the same task. The two plots below compare the performance of the machine and the human. Rewards are calculated based on how close the vehicle flies from the obstacles.

[![VIDEO: Autonomous Collision Avoidance for UAVs in AirSim](https://img.youtube.com/vi/rEbNytWz3c8/0.jpg)](https://www.youtube.com/watch?v=rEbNytWz3c8)

## Installation

You have to setup a few things before have everything running:  

1. [Microsoft AirSim](https://github.com/Microsoft/AirSim). Currently, it must be installed on Windows. Since all the learning runs on Ubuntu, you also have to install Bash on Windows (Pixhawk firmware 1.4.4).  

2. [Bash on Windows](https://msdn.microsoft.com/en-us/commandline/wsl/install_guide). Microsoft is developing a Ubuntu shell inside Windows that is running fairly well (besides a few drawbacks not really important at this point). After you installed it you should be able to open a terminal and install packages via pip, sudo-apt, etc, as you were using a Linux/Ubuntu machine.   

3. Install [Xming Xlauncher](https://sourceforge.net/projects/xming/) so your Ubuntu shell can open new windows (like when you plot using matplotlib or create a new window using OpenCV). Then enable the Ubuntu shell to use the display and create new windows:
```
echo 'export DISPLAY=:0' >> ~/.bashrc
dbus-uuidgen > /etc/machine-id
source ~/.bashrc
```  

4. Follow the instructions bellow to install the learning libraries.  

### Installation: Pixhawk

Download [QGroundControl](http://qgroundcontrol.com/downloads/), download the [PX4 Firmware v1.4.4](https://github.com/PX4/Firmware/releases/download/v1.4.4/Firmware.zip). Update your Pixhawk with the PX4 firmware and load the parameters from the [arl_px4](./arl_px4) file in this folder (QGroundControl > Parameters > Tools > Load parameters).

### Installation: Learning Libraries

All the dependencies for the learning code are being handled by Anaconda3. You should be able to just run the setup script to install everything you might need. You can check [this file](./environment.yml) to see all libraries that will be installed.

Please go ahead and [install Anaconda3 for LINUX](https://www.continuum.io/downloads#linux) following these instructions:  
**NOTE:** Accept the default location and answer YES when asked about preped Anaconda3 install location.  
```
wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
bash https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
source ~/.bashrc
```

Clone this repository. Go to its main folder and run:
```
cd arl/
sed -i 's/\r//' setup_linux.sh
./setup_linux.sh
```

Need to fix OpenCV installation because Windows is awesome:  
```
conda install -c menpo opencv3=3.1.0
sudo apt-get install execstack
sudo execstack -c $HOME/anaconda3/envs/arl/lib/libopencv_*
sudo apt-get install gtk2.0-0
```
**NOTE:** If you have any other error related to the OpenCV installation, please check these links: [link 1](http://tatsuya-y.hatenablog.com/entry/2016/08/29/183331) and [link 2](http://www.pcworld.com/article/3055403/windows/windows-10s-bash-shell-can-run-graphical-linux-applications-with-this-trick.html).

All dependencies should be installed by now. Let's activate the environment and run the test script:
```
source activate arl
./test/test_env.py
```

**NOTE:** Currently, Bash On Windows [does not support](https://github.com/Microsoft/BashOnWindows/issues/1788) using the GPU-supported version of Tensorflow.

The script will import all modules and report their versions. If there's any problem, please contact Vinicius Guimaraes Goecks at vinicius.goecks@tamu.edu.


Remember to always activate the ARL Anaconda Environment before using it (in case you close the terminal):
```
source activate arl
```

You might want to deactivate it eventually:
```
source deactivate
```

**NOTE:** If you are under a Linux environment you can use the GPU version of Tensorflow. First, uninstall the CPU version of Tensorflow:
```
pip uninstall tensorflow
```
This [webpage](https://medium.com/@ikekramer/installing-cuda-8-0-and-cudnn-5-1-on-ubuntu-16-04-6b9f284f6e77) will guide you through the Tensorflow-GPU installation process.

## Testing the System

Make sure AirSim is compiled and you followed all the steps on the [Microsoft AirSim](https://github.com/Microsoft/AirSim) page. Assuming you have Unreal Engine and AirSim installed on your Unreal Project, you are just a few steps of having an autonomous flying quadrotor.

1. Run the XLaunch, so the Ubuntu shell can open additional windows if necessary.  
2. Open a Windows Command Window, type 'bash' and press enter. Now you are on the Ubuntu shell. Navigate to the "arl/arl" folder.  
3. Go to Unreal Projects in Documents, double click on the .sln file to open Visual Studio. Once it is open, press Ctrl+F5 to run Unreal Engine without the debug mode.  
4. Make sure you have the Pixhawk connected or the SITL version of it (as explained on the [SITL AirSim page](https://github.com/Microsoft/AirSim/blob/master/docs/sitl.md)) on a separated Ubuntu terminal.  
5. Press Alt+P on Unreal Engine to spawn the quadrotor.  
6. Run the main code:

```
python main.py
```

The quad should takeoff and flight autonomously along the path.

# Configurating the Main File

To be written.

# Known Issues

1. Had problems running it on Windows 10 Build16241.rs_prerelease.170708-1800 (Insider Program): error on tornado pacakge.
