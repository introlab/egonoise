#!/usr/bin/env python3

import sys
import os
import time

import numpy as np

sys.path.append('/home/pierre-olivier/Git/kissdsp')

import rospy
import rosbag

from audio_utils.msg import AudioFrame
from audio_utils import get_format_information, convert_audio_data_to_numpy_frames, convert_numpy_frames_to_audio_data
from utils.egonoise_utils import *


class EgoNoiseRun:
    def __init__(self):
        self._input_format = rospy.get_param('~input_format', '')
        self._output_format = rospy.get_param('~output_format', '')
        self._dict_path = rospy.get_param('~dict_path', '')
        self._frame_size = rospy.get_param('~frame_size', '')
        self._channel_keep = rospy.get_param('~channel_keep', '')
        self._bag_noise = rospy.get_param('~bag_noise', '')
        self._bag_speech = rospy.get_param('~bag_speech', '')
        self._publish = rospy.get_param('~publish', '')

        self._input_format_information = get_format_information(self._input_format)
        self._output_format_information = get_format_information(self._output_format)

        self._audio_frame_msg = AudioFrame()

        self.RRs_dict, self.RRs_inv_dict = load_dictionnary(self._dict_path, self._frame_size)

        if self._publish:
            self._audio_frame_msg = AudioFrame()
            self._audio_pub = rospy.Publisher('audio_in', AudioFrame, queue_size=10)

        self.last_window = np.zeros((len(self._channel_keep), int(1.5*self._frame_size)))

    def run(self):
        for (_, msg_speech, _), (_, msg_noise, _) in zip(rosbag.Bag(self._bag_speech).read_messages(), rosbag.Bag(self._bag_noise).read_messages()):
            frames_speech = np.array(convert_audio_data_to_numpy_frames(self._input_format_information, msg_speech.channel_count, msg_speech.data))[self._channel_keep]
            frames_noise = np.array(convert_audio_data_to_numpy_frames(self._input_format_information, msg_noise.channel_count, msg_noise.data))[self._channel_keep]
            frames = frames_speech + frames_noise
            frames = np.hstack((self.last_window, frames))
            self.last_window = frames[:, -int(1.5*self._frame_size):]
            frame_cleaned = egonoise(frames, self.RRs_dict, self.RRs_inv_dict,
                                     self._frame_size, len(self._channel_keep),
                                     verbose=True, frames_speech=frames_speech, frames_noise=frames_noise)
            frame_cleaned = frame_cleaned[:, int(0.75*self._frame_size):-int(0.75*self._frame_size)]
            print(frame_cleaned.shape)

            if self._publish:
                data = convert_numpy_frames_to_audio_data(self._output_format_information, frame_cleaned)

                self._audio_frame_msg.header = msg_speech.header
                self._audio_frame_msg.format = self._output_format
                self._audio_frame_msg.channel_count = 1
                self._audio_frame_msg.sampling_frequency = msg_speech.sampling_frequency
                self._audio_frame_msg.frame_sample_count = msg_speech.frame_sample_count
                self._audio_frame_msg.data = data

                self._audio_pub.publish(self._audio_frame_msg)
                # self._audio_pub.publish(msg_speech)


def main():
    rospy.init_node('egonoise_run', log_level=rospy.DEBUG)
    egonoise_run = EgoNoiseRun()
    egonoise_run.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
