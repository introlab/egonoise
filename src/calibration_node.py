#!/usr/bin/env python3

import rospy

from audio_utils.msg import AudioFrame
from audio_utils import get_format_information, convert_audio_data_to_numpy_frames
from utils.calibration_utils import *


class CalibrationNode:
    def __init__(self):
        self._input_format = rospy.get_param('~input_format', '')
        self._database_path = rospy.get_param('~database_path', '')
        self._sampling_frequency = rospy.get_param('~sampling_frequency', '')
        self._frame_sample_count = rospy.get_param('~frame_sample_count', '')
        self._frame_size = rospy.get_param('~frame_size', '')
        self._channel_count = rospy.get_param('~channel_count', '')
        self._overlap = rospy.get_param('~overlap', '')
        self._hop_length = rospy.get_param('~hop_length', '')
        self._calibration_duration = rospy.get_param('~calibration_duration', '')
        self._calibration_step = rospy.get_param('~calibration_step', '')
        self._n_frame_scm = rospy.get_param('~n_frame_scm', '')

        self._input_format_information = get_format_information(self._input_format)

        self._accumulation_frame = np.array([], dtype=np.float32).reshape(self._channel_count, 0)
        self._len_window = int(self._n_frame_scm  * self._hop_length)

        self._idx = 0
        self._tfs  = []

        self._nb_calibration_examples = int(self._calibration_duration*self._sampling_frequency/self._calibration_step)

        reset_database(self._database_path)

        self._audio_sub = rospy.Subscriber('audio_out', AudioFrame, self._audio_cb, queue_size=100)

    def _audio_cb(self, msg):
        if self._nb_calibration_examples > self._idx :
            if msg.format != self._input_format:
                rospy.logerr('Invalid input format (msg.format={}, param.input_format={})'.format(msg.format, self._input_format))
                return

            frames = np.array(convert_audio_data_to_numpy_frames(self._input_format_information, msg.channel_count, msg.data))
            self._accumulation_frame = np.hstack((self._accumulation_frame, frames))

            if self._accumulation_frame.shape[1]>self._len_window:
                while (self._idx*self._calibration_step+self._len_window)<self._accumulation_frame.shape[1]:
                    window = self._accumulation_frame[:, self._idx*self._calibration_step:(self._idx*self._calibration_step + self._len_window)]
                    tf = save_scm(window, f'{self._database_path}{self._idx}', self._frame_size, self._hop_length)
                    self._tfs.append(tf)
                    self._idx += 1

        else:
            save_pca(self._tfs, self._database_path)
            rospy.loginfo('Calibration is finished!')

    def run(self):
        rospy.spin()


def main():
    rospy.init_node('egonoise_node', log_level=rospy.INFO)
    calibration_node = CalibrationNode()
    calibration_node.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException or SystemExit:
        pass
