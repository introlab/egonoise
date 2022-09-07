#!/usr/bin/env python3

import sys
sys.path.append('/home/pierre-olivier/Git/kissdsp')

import numpy as np

import rospy

from audio_utils.msg import AudioFrame
from audio_utils import get_format_information, convert_audio_data_to_numpy_frames, convert_numpy_frames_to_audio_data
import kissdsp.sink as snk


class CalibrationNode:
    def __init__(self):
        self._input_format = rospy.get_param('~input_format', '')
        self._database_path = rospy.get_param('~database_path', '')
        self._channel_keep = rospy.get_param('~channel_keep', '')

        self._input_format_information = get_format_information(self._input_format)

        self._audio_frame_msg = AudioFrame()

        self._audio_sub = rospy.Subscriber('audio_out', AudioFrame, self._audio_cb, queue_size=10)

        self.idx = 0


    def _audio_cb(self, msg):
        if msg.format != self._input_format:
            rospy.logerr('Invalid input format (msg.format={}, param.input_format={})'.format(msg.format, self._input_format))
            return

        frames = convert_audio_data_to_numpy_frames(self._input_format_information, msg.channel_count, msg.data)
        frames = np.array(frames)[self._channel_keep]

        snk.write(frames, f'{self._database_path}{self.idx}.wav', msg.sampling_frequency)

        self.idx += 1

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('calibration_node', log_level=rospy.DEBUG)
    calibration_node = CalibrationNode()
    calibration_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
