#!/usr/bin/env python3

import rospy

from audio_utils.msg import AudioFrame
from audio_utils import get_format_information, convert_audio_data_to_numpy_frames, convert_numpy_frames_to_audio_data
from utils.egonoise_utils import *

import kissdsp.filterbank as fb
import kissdsp.spatial as sp

from time import time


class EgoNoiseNode:
    def __init__(self):
        self._input_format = rospy.get_param('~input_format', '')
        self._output_format = rospy.get_param('~output_format', '')
        self._database_path = rospy.get_param('~database_path', '')
        self._frame_size = rospy.get_param('~frame_size', '')
        self._channel_count = rospy.get_param('~channel_count', '')
        self._overlap = rospy.get_param('~overlap', '')
        self._hop_length = rospy.get_param('~hop_length', '')

        self._input_format_information = get_format_information(self._input_format)
        self._output_format_information = get_format_information(self._output_format)

        self._audio_frame_msg = AudioFrame()

        self._pca, self._pca_dict = load_pca(self._database_path)

        self._audio_pub = rospy.Publisher('audio_in', AudioFrame, queue_size=10)
        self._audio_sub = rospy.Subscriber('audio_out', AudioFrame, self._audio_cb, queue_size=10)

        self._n_batch = 6
        self._overlap = 1.5
        self._frames = np.ones((self._channel_count, int(self._overlap*self._frame_size+self._n_batch*self._hop_length)))
        self._list_YYs = []
        self._Zs = None
        self._seq = 0
        self._zs = []


    def _audio_cb(self, msg):
        if msg.format != self._input_format:
            rospy.logerr('Invalid input format (msg.format={}, param.input_format={})'.format(msg.format, self._input_format))
            return

        time0 = time()

        frames = np.array(convert_audio_data_to_numpy_frames(self._input_format_information, msg.channel_count, msg.data))

        self._frames[:, :-self._hop_length] = self._frames[:, self._hop_length:]
        self._frames[:, -self._hop_length:] = frames

        self._seq += 1

        if len(self._zs)>0:
            zs = self._zs.pop(0)[None,...]
            data = convert_numpy_frames_to_audio_data(self._output_format_information, zs)

            self._audio_frame_msg.header = msg.header
            self._audio_frame_msg.format = self._output_format
            self._audio_frame_msg.channel_count = 1
            self._audio_frame_msg.sampling_frequency = msg.sampling_frequency
            self._audio_frame_msg.frame_sample_count = msg.frame_sample_count
            self._audio_frame_msg.data = data

            self._audio_pub.publish(self._audio_frame_msg)

        print(f'CallBack Time: {time() - time0}')

    def noise_reduction(self):
        if self._seq>self._n_batch:
            time0 = time()

            Ys = fb.stft(self._frames, frame_size=self._frame_size, hop_size=self._hop_length)
            YYs = sp.scm(sp.xspec(Ys))

            if len(self._list_YYs)==0:
                self._YYs = YYs
                self._list_YYs.append(YYs)
            elif len(self._list_YYs)<self._n_batch:
                self._YYs += YYs
                self._list_YYs.append(YYs)
            else:
                self._YYs += YYs
                self._YYs -= self._list_YYs.pop(0)
                self._list_YYs.append(YYs)

            # PCA
            val = compute_pca(YYs, self._pca)
            diff = np.sum(abs(val - self._pca_dict), axis=1)
            idx = np.argmin(diff)
            RRsInv = load_scm(self._database_path, idx, self._frame_size, len(self._frames))

            # MVDR
            Zs, ws = compute_mvdr(Ys, YYs, RRsInv)

            zs = fb.istft(Zs, hop_size=self._hop_length)
            self._zs = list(zs[:, int(self._overlap/2*self._frame_size):-int(self._overlap/2*self._frame_size)].reshape(self._n_batch,  self._hop_length))
            # print(list(self._zs))

            print(f'Reduction Time: {time() - time0}')





def main():
    rospy.init_node('egonoise_node', log_level=rospy.INFO)
    egonoise_node = EgoNoiseNode()

    rospy.sleep(1)

    r = rospy.Rate((32000/(egonoise_node._n_batch*256)))

    while not rospy.is_shutdown():
        egonoise_node.noise_reduction()
        r.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
