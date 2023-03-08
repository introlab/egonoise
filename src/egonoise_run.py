#!/usr/bin/env python3

import rospy
import rosbag
import time

from audio_utils.msg import AudioFrame
from audio_utils import get_format_information, convert_audio_data_to_numpy_frames, convert_numpy_frames_to_audio_data
from utils.egonoise_utils import *

import kissdsp.filterbank as fb
import kissdsp.spatial as sp


class EgoNoiseRun:
    def __init__(self):
        self._input_format = rospy.get_param('~input_format', '')
        self._output_format = rospy.get_param('~output_format', '')
        self._database_path = rospy.get_param('~database_path', '')
        self._frame_size = rospy.get_param('~frame_size', '')
        self._channel_count = rospy.get_param('~channel_count', '')
        self._bag_noise = rospy.get_param('~bag_noise', '')
        self._bag_speech = rospy.get_param('~bag_speech', '')
        self._overlap = rospy.get_param('~overlap', '')
        self._hop_length = rospy.get_param('~hop_length', '')

        self._input_format_information = get_format_information(self._input_format)
        self._output_format_information = get_format_information(self._output_format)

        self._audio_frame_msg = AudioFrame()
        self._audio_pub = rospy.Publisher('audio_in', AudioFrame, queue_size=10)

        self.pca, self.pca_dict = load_pca(self._database_path)

        self.last_window = np.zeros((self._channel_count, int(self._overlap*self._frame_size)))
        self.last_window_s = np.zeros((self._channel_count, int(self._overlap * self._frame_size)))
        self.last_window_n = np.zeros((self._channel_count, int(self._overlap*self._frame_size)))

        self.istft_cut = int((self._overlap / 2) * self._frame_size / self._hop_length)


    def run(self):
        for (_, msg_speech, _), (_, msg_noise, _) in zip(rosbag.Bag(self._bag_speech).read_messages(), rosbag.Bag(self._bag_noise).read_messages()):
            frames_speech = np.array(convert_audio_data_to_numpy_frames(self._input_format_information, msg_speech.channel_count, msg_speech.data))
            frames_noise = np.array(convert_audio_data_to_numpy_frames(self._input_format_information, msg_noise.channel_count, msg_noise.data))

            frames = frames_noise + frames_speech
            frames = np.hstack((self.last_window, frames))

            frames_speech = np.hstack((self.last_window_s, frames_speech))
            frames_noise = np.hstack((self.last_window_n, frames_noise))

            self.last_window = frames[:, -int(self._overlap*self._frame_size):]
            self.last_window_s = frames_speech[:, -int(self._overlap*self._frame_size):]
            self.last_window_n = frames_noise[:, -int(self._overlap*self._frame_size):]

            # STFT and SCM
            Ys = fb.stft(frames, frame_size=self._frame_size, hop_size=self._hop_length)
            YYs = sp.scm(sp.xspec(Ys))

            # PCA
            val = compute_pca(YYs, self.pca)
            diff = np.sum(abs(val - self.pca_dict), axis=1)
            idx = np.argmin(diff)
            RRsInv = load_scm(self._database_path, idx, self._frame_size, len(frames))

            # MVDR
            Zs, ws = compute_mvdr(Ys, YYs, RRsInv)

            # ISTFT
            zs = fb.istft(Zs, hop_size=self._hop_length)[:, self.istft_cut:-self.istft_cut]

            data = convert_numpy_frames_to_audio_data(self._output_format_information, zs)

            self._audio_frame_msg.header = msg_speech.header
            self._audio_frame_msg.format = self._output_format
            self._audio_frame_msg.channel_count = 1
            self._audio_frame_msg.sampling_frequency = msg_speech.sampling_frequency
            self._audio_frame_msg.frame_sample_count = msg_speech.frame_sample_count
            self._audio_frame_msg.data = data

            self._audio_pub.publish(self._audio_frame_msg)


def main():
    rospy.init_node('egonoise_run', log_level=rospy.INFO)
    egonoise_run = EgoNoiseRun()
    egonoise_run.run()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
