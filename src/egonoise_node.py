#!/usr/bin/env python3

import sys
sys.path.append('/home/pierre-olivier/Git/kissdsp')

import rospy

from audio_utils.msg import AudioFrame
from audio_utils import get_format_information, convert_audio_data_to_numpy_frames, convert_numpy_frames_to_audio_data
from utils.egonoise_utils import *


class EgoNoiseNode:
    def __init__(self):
        self._input_format = rospy.get_param('~input_format', '')
        self._output_format = rospy.get_param('~output_format', '')
        self._dict_path = rospy.get_param('~dict_path', '')
        self._frame_size = rospy.get_param('~frame_size', '')
        self._channel_keep = rospy.get_param('~channel_keep', '')
        self._overlap = rospy.get_param('~overlap', '')
        self._hop_length = rospy.get_param('~hop_length', '')

        self._input_format_information = get_format_information(self._input_format)
        self._output_format_information = get_format_information(self._output_format)

        self._audio_frame_msg = AudioFrame()

        self.RRs_dict, self.RRs_inv_dict = load_dictionnary(self._dict_path, self._frame_size, self._hop_length)

        self._audio_pub = rospy.Publisher('audio_in', AudioFrame, queue_size=10)
        self._audio_sub = rospy.Subscriber('audio_out', AudioFrame, self._audio_cb, queue_size=10)

        self.last_window = np.zeros((len(self._channel_keep), int(self._overlap*self._frame_size)))

    def _audio_cb(self, msg):
        if msg.format != self._input_format:
            rospy.logerr('Invalid input format (msg.format={}, param.input_format={})'.format(msg.format, self._input_format))
            return

        frames = np.array(convert_audio_data_to_numpy_frames(self._input_format_information, msg.channel_count, msg.data))[self._channel_keep]
        frames = np.hstack((self.last_window, frames))
        self.last_window = frames[:, -int(self._overlap * self._frame_size):]
        frame_cleaned = egonoise(frames, self.RRs_dict, self.RRs_inv_dict, self._frame_size, len(self._channel_keep), self._hop_length)
        frame_cleaned = frame_cleaned[:, int(self._overlap / 2 * self._frame_size):-int(self._overlap / 2 * self._frame_size)]
        data = convert_numpy_frames_to_audio_data(self._output_format_information, frame_cleaned)

        self._audio_frame_msg.header = msg.header
        self._audio_frame_msg.format = self._output_format
        self._audio_frame_msg.channel_count = 1
        self._audio_frame_msg.sampling_frequency = msg.sampling_frequency
        self._audio_frame_msg.frame_sample_count = msg.frame_sample_count
        self._audio_frame_msg.data = data

        self._audio_pub.publish(self._audio_frame_msg)

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('egonoise_node', log_level=rospy.DEBUG)
    egonoise_node = EgoNoiseNode()
    egonoise_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
