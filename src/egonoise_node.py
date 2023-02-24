#!/usr/bin/env python3

import rospy
from time import time

from audio_utils.msg import AudioFrame
from audio_utils import get_format_information, convert_audio_data_to_numpy_frames, convert_numpy_frames_to_audio_data
from utils.egonoise_utils import *

import kissdsp.filterbank as fb
import kissdsp.spatial as sp


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

        self._pca, self._pca_dict = load_pca(self._dict_path)

        self._audio_pub = rospy.Publisher('audio_in', AudioFrame, queue_size=10)
        self._audio_sub = rospy.Subscriber('audio_out', AudioFrame, self._audio_cb, queue_size=10)

        self._last_window = np.zeros((len(self._channel_keep), int(self._overlap*self._frame_size)))

        self._istft_cut =  int(self._overlap / 2 * self._frame_size)


    def _audio_cb(self, msg):
        n = time()
        if msg.format != self._input_format:
            rospy.logerr('Invalid input format (msg.format={}, param.input_format={})'.format(msg.format, self._input_format))
            return

        frames = np.array(convert_audio_data_to_numpy_frames(self._input_format_information, msg.channel_count, msg.data))[self._channel_keep]

        frames = np.hstack((self._last_window, frames))

        self._last_window = frames[:, -int(self._overlap * self._frame_size):]

        # STFT and SCM
        Ys = fb.stft(frames, frame_size=self._frame_size, hop_size=self._hop_length)
        YYs = sp.scm(sp.xspec(Ys))

        # PCA
        val = compute_pca(YYs, self._pca)
        diff = np.sum(abs(val - self._pca_dict), axis=1)
        idx = np.argmin(diff)
        RRsInv = load_scm(self._dict_path, idx, self._frame_size, len(frames))

        # MVDR
        Zs, ws = compute_mvdr(Ys, YYs, RRsInv)

        # ISTFT
        zs = fb.istft(Zs, hop_size=self._hop_length)[:, self._istft_cut:-self._istft_cut]

        data = convert_numpy_frames_to_audio_data(self._output_format_information, zs)

        self._audio_frame_msg.header = msg.header
        self._audio_frame_msg.format = self._output_format
        self._audio_frame_msg.channel_count = 1
        self._audio_frame_msg.sampling_frequency = msg.sampling_frequency
        self._audio_frame_msg.frame_sample_count = msg.frame_sample_count
        self._audio_frame_msg.data = data

        self._audio_pub.publish(self._audio_frame_msg)
        a = time()
        print(f'Time: {a-n}')

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
