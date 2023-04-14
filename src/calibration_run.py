#!/usr/bin/env python3

import rospy

from audio_utils import get_format_information
from utils.calibration_utils import calibration_run


class CalibrationRun:
    def __init__(self):
        self._input_format = rospy.get_param('~input_format', '')
        self._database_path = rospy.get_param('~database_path', '')
        self._bag_calibration = rospy.get_param('~bag_calibration', '')
        self._bag_calibration_path = rospy.get_param('~bag_calibration_path', '')
        self._frame_size = rospy.get_param('~frame_size', '')
        self._hop_length = rospy.get_param('~hop_length', '')
        self._overlap = rospy.get_param('~overlap', '')
        self._calibration_step = rospy.get_param('~calibration_step', '')
        self._n_frame_scm = rospy.get_param('~n_frame_scm', '')


        self._list_bag_calibration = self._bag_calibration.split(',')
        self._input_format_information = get_format_information(self._input_format)

    def run(self):
        calibration_run(self._bag_calibration_path, self._list_bag_calibration, self._frame_size, self._hop_length, self._overlap,
                        self._input_format_information, self._database_path, self._n_frame_scm, self._calibration_step)


def main():
    rospy.init_node('calibration_run', log_level=rospy.INFO)
    calibration_run = CalibrationRun()
    calibration_run.run()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
