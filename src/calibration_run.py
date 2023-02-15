#!/usr/bin/env python3

import rospy
import shutil

from audio_utils import get_format_information
from utils.calibration_utils import save_db
from utils import list_info


class CalibrationRun:
    def __init__(self):
        self._input_format = rospy.get_param('~input_format', '')
        self._database_path = list_info.dict_path #rospy.get_param('~database_path', '')
        self._channel_keep = rospy.get_param('~channel_keep', '')
        self._bag_path = list_info.bag_path #rospy.get_param('~bag_name', '')
        self._frame_size = rospy.get_param('~frame_size', '')
        self._overlap = rospy.get_param('~overlap', '')
        self._overwrite_dict = rospy.get_param('~overwrite_dict', '')

        self._input_format_information = get_format_information(self._input_format)

    def run(self):
        save_db(self._bag_path, self._channel_keep, self._frame_size, self._overlap, self._input_format_information, self._database_path)


def main():
    rospy.init_node('calibration_run', log_level=rospy.INFO)
    calibration_run = CalibrationRun()
    calibration_run.run()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
