# egonoise

The goal of this project is to create a system able to reduce easily the egonoise of a robot using the Minimum Variance Distortionless Response (MVDR) algorithm.

Author(s): Pierre-Olivier Lagacé

**  This Repo is in progress

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

## Custom Python library
1. Install https://github.com/FrancoisGrondin/kissdsp

### calibration_run.py
Expication: This node allow to train de database with a rosbag using the command `roslaunch egonoise egonoise.launch calibration_run:=true`.
Parameters:
 - input_format
 - sampling_frequency
 - database_path
 - bag_name
 - frame_size
 - overlap
 - channel_keep
 - device

### calibration_node.py TODO
Expication: This node allow to train de database with live input using the command `roslaunch egonoise egonoise.launch calibration_node:=true`.
Parameters:
 - ...
Topics (Sub and Pub)
 - audio_out


### egonoise_node.py
Expication: This node allow to use the framework to filtered the signal using the command `roslaunch egonoise egonoise.launch egonoise_node:=true`
Parameters:
 - input_format
 - output_format
 - dict_path
 - frame_size
 - channel_count
 - overlap
 - hop_length
 - channel_keep
Topics (Sub and Pub)
 - audio_out
 - audio_in

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
- Install pulse_audio utils
- if `pactl list` do `pa_context_connect() failed: Connection refused` try `sudo apt-get --purge --reinstall install pulseaudio`
- sudo apt-get install libportaudio2
- Test a python script to make a test record
8. Install Ros Noetic
- Use `catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3` for the first catkin_make
9 Install KissDsp
10. Install this project in catkin_ws/src/
11. Install audio_utils
12. Test with: `roslaunch egonoise egonoise.launch audio_capture:=true` with the good parameter.
13. Follow the guide https://husarion.com/tutorials/ros-tutorials/5-running-ros-on-multiple-machines/ if you want to record the rosbag on another machine.

