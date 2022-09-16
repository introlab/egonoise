#!/usr/bin/env python3

import rosbag
import numpy as np

from audio_utils import get_format_information, convert_audio_data_to_numpy_frames, convert_numpy_frames_to_audio_data
from audio_utils.msg import AudioFrame

bag_name1 = 'noise4mics'
bag_name2 = 'speech4mics'

bag_path1 = f'/home/pierre-olivier/catkin_ws/src/bag/{bag_name1}.bag'
bag_path2 = f'/home/pierre-olivier/catkin_ws/src/bag/{bag_name2}.bag'

bag_path_out = f'/home/pierre-olivier/catkin_ws/src/bag/{bag_name1}--{bag_name2}.bag'

audio_frame_msg = AudioFrame()

with rosbag.Bag(bag_path_out, 'w') as outbag:
    for (topic1, msg1, _), (_, msg2, _) in zip(rosbag.Bag(bag_path1).read_messages(), rosbag.Bag(bag_path2).read_messages()):
        if msg1.format != msg2.format \
                or msg1.channel_count != msg2.channel_count \
                or msg1.sampling_frequency != msg2.sampling_frequency:
            print('Error: Bag are not compatible')
            break

        frames1 = np.array(convert_audio_data_to_numpy_frames(get_format_information(msg1.format),
                                                                    msg1.channel_count, msg1.data))
        frames2 = np.array(convert_audio_data_to_numpy_frames(get_format_information(msg2.format),
                                                              msg2.channel_count, msg2.data))
        frames = frames1 + frames2

        data = convert_numpy_frames_to_audio_data(get_format_information(msg1.format), frames)

        msg1.data = data

        outbag.write(topic1, msg1, msg1.header.stamp)