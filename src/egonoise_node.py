#!/usr/bin/env python3

import sys
import os
import time
sys.path.append('/home/pierre-olivier/Git/kissdsp')

import numpy as np

import rospy
import logging

from audio_utils.msg import AudioFrame
from audio_utils import get_format_information, convert_audio_data_to_numpy_frames, convert_numpy_frames_to_audio_data
import kissdsp.source as src
import kissdsp.filterbank as fb
import kissdsp.spatial as sp
import kissdsp.beamformer as bf


class EgoNoiseNode:
    def __init__(self):
        self._input_format = rospy.get_param('~input_format', '')
        self._output_format = rospy.get_param('~output_format', '')
        self._dict_path = rospy.get_param('~dict_path', '')
        self._frame_size = rospy.get_param('~frame_size', '')
        self._channel_keep = rospy.get_param('~channel_keep', '')

        self._input_format_information = get_format_information(self._input_format)
        self._output_format_information = get_format_information(self._output_format)

        self._audio_frame_msg = AudioFrame()

        now = time.time_ns()
        self.RRs_dict, self.RRs_inv_dict = self.load_dictionnary(self._dict_path, self._frame_size)
        rospy.loginfo(f'Dictionnary import time: {(time.time_ns()-now)/1e9} seconds')
        rospy.loginfo(f'Dictionnary dimension: {self.RRs_dict.shape}')

        self._audio_pub = rospy.Publisher('audio_in', AudioFrame, queue_size=10)
        self._audio_sub = rospy.Subscriber('audio_out', AudioFrame, self._audio_cb, queue_size=10)

    def load_dictionnary(self, path, frame_size):
        files = os.listdir(path)
        RRs_list = []
        for file in files:
            p = f'{path}/{file}'
            wav = src.read(p)
            Rs = fb.stft(wav, frame_size=frame_size)
            RRs = sp.scm(Rs)
            RRs_list.append(RRs)
        RRs_dict = np.stack(RRs_list)
        RRs_dict_inv = np.linalg.inv(RRs_dict)
        return RRs_dict, RRs_dict_inv

    def egonoise(self, frames, RRs_dict, RRs_inv_dict, frame_size, _channel_keep):
        Ys = fb.stft(frames, frame_size=frame_size)
        YYs = sp.scm(Ys)
        diff = np.sum(abs(RRs_inv_dict@(YYs-RRs_dict)-np.eye(_channel_keep)))
        idx = np.argmin(diff)
        RRs = RRs_dict[idx]
        TTs = YYs - RRs
        vs = sp.steering(TTs)  # Compute steering vector
        ws = bf.mvdr(vs, RRs)  # Compute mvdr weights
        Zs = bf.beam(Ys, ws)  # Perform beamforming
        zs = fb.istft(Zs) # Return to time domain
        return zs

    def _audio_cb(self, msg):
        if msg.format != self._input_format:
            rospy.logerr('Invalid input format (msg.format={}, param.input_format={})'.format(msg.format, self._input_format))
            return

        now = time.time_ns()
        frames = convert_audio_data_to_numpy_frames(self._input_format_information, msg.channel_count, msg.data)
        frame_cleaned = self.egonoise(np.array(frames)[self._channel_keep], self.RRs_dict, self.RRs_inv_dict, self._frame_size, len(self._channel_keep))
        data = convert_numpy_frames_to_audio_data(self._output_format_information, frame_cleaned)
        rospy.loginfo(f'Processing time: {(time.time_ns() - now) / 1e9} seconds')

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
