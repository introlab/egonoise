#!/usr/bin/env python3

import sys
sys.path.append('/home/pierre-olivier/Git/kissdsp')

import numpy as np

import rospy
import  rosbag

from audio_utils.msg import AudioFrame
from audio_utils import get_format_information, convert_audio_data_to_numpy_frames, convert_numpy_frames_to_audio_data
import kissdsp.sink as snk


class MergingNode:
    def __init__(self):
        self._input_format = rospy.get_param('~format', '')

        self._input_format_information = get_format_information(self._input_format)

        self._audio_frame_msg = AudioFrame()

        self._audio_speech_sub = rospy.Subscriber('audio_speech', AudioFrame, self._audio_cb, queue_size=10)
        self._audio_noise_sub = rospy.Subscriber('audio_noise', AudioFrame, self._audio_cb, queue_size=10)
        self._audio_pub = rospy.Publisher('audio_out', AudioFrame, queue_size=10)


        self.idx = 0


    def _audio_cb(self, msg):
        if msg.format != self._input_format:
            rospy.logerr('Invalid input format (msg.format={}, param.input_format={})'.format(msg.format, self._input_format))
            return

        frames = convert_audio_data_to_numpy_frames(self._input_format_information, msg.channel_count, msg.data)
        frames = np.array(frames)


    def run(self):
        rospy.spin()


def main():
    rospy.init_node('merging_node', log_level=rospy.DEBUG)
    merging_node = MergingNode()
    merging_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
