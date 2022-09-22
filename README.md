# egonoise

The goal of this project is to create a system able to reduce easily the egonoise of a robot using the Minimum Variance Distortionless Response (MVDR) algorithm.

Author(s): Pierre-Olivier Lagacé

## Installation

### ROS Noetic on Ubuntu 20.04
1. Follow the instructions on the offical website ([ROS Installation](http://wiki.ros.org/noetic/Installation/Ubuntu))
2. If you are not familiar with ROS, we strongly recommend that you do the tutorials ([ROS Tutorials](http://wiki.ros.org/ROS/Tutorials))

### Configuring your Catkin Workspace and Installation
1. Make shure that the end of your `.bashrc` file in your `home` folder has the following lines
```
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

### Python
1. Install Python 3.10.5
2. Clone this repo in your catkin_ws/src
3. Install the requirement.txt 
```
pip3 install -r requirement.txt
```
4. Clone the repo https://github.com/FrancoisGrondin/kissdsp on your computer (not in the catkin_ws) and test it.
5. Add this to the end of your `bashrc`
```
export PYTHONPATH="${PYTHONPATH}:"/<kissdsp_path>/kissdsp"
```

### ROS Libraries
1. Install https://github.com/introlab/audio_utils in your catkin_ws/src

## Scripts
TODO

### calibration_run.py
Expication:
Parameters:
Topics (Sub and Pub)

### egonoise_run.py
Expication:
Parameters:
Topics (Sub and Pub)

### egonoise_node.py
Expication:
Parameters:
Topics (Sub and Pub)

### bag2wav.py
Expication:
Parameters:
Topics (Sub and Pub)

### bag_merge.py
Expication:
Parameters:
Topics (Sub and Pub)


## Setup RaspberryPi
### Info
username: ubuntu
password: egonoise

### Installation
1. Flash SD card with ubuntu 20.04 server 64bits using RaspberryPi imager
2. Launch RaspberryPi with SD card
3. Setup Wifi -> https://linuxconfig.org/ubuntu-20-04-connect-to-wifi-from-command-line
5. sudo apt-get update (and upgrade?)
6. Install Python: Pyenv -> https://k0nze.dev/posts/install-pyenv-venv-vscode
7. Tests with microphones array:
- Install dependency (Alsa and pulseaudio)
- Test a python script to make a test record
8. Install Ros Noetic
9. Install this project in catkin_ws/src/
10. Install audio_utils
11. Test
TODO: Need to improve the step 

