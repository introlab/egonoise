#!/usr/bin/env python3

import rosbag
import rospy

from audio_utils import get_format_information
from utils.calibration_utils import save_db


class CalibrationRun:
    def __init__(self):
        self._input_format = rospy.get_param('~input_format', '')
        self._database_path = rospy.get_param('~database_path', '')
        self._channel_keep = rospy.get_param('~channel_keep', '')
        self._bag_path = rospy.get_param('~bag_name', '')

        self._input_format_information = get_format_information(self._input_format)

        # TODO
        # Parameter to delete current database?
        # Egonoise during calibration?

    def run(self):
        for idx, (_, msg, _) in enumerate(rosbag.Bag(self._bag_path).read_messages()):
            save_db(msg, idx, self._channel_keep, self._input_format_information, self._database_path)


def main():
    rospy.init_node('calibration_run', log_level=rospy.DEBUG)
    calibration_run = CalibrationRun()
    calibration_run.run()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
