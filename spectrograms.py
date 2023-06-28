# -*- Coding:utf-8 -*-
# Author:Universe Feng

import librosa
import numpy as np
import matplotlib.pyplot as plt

class MySpec(object):
    def __init__(self,filename,wlen,inc):
        #参数预定义
        self.filename=filename
        self.wlen=wlen
        self.inc=inc
        self.nfft=self.wlen
        self.win=self.hanning_window(self.wlen)
    # 计算每帧对应的时间
    def FrameTimeC(self,frameNum, frameLen, inc, fs):
        ll = np.array([i for i in range(frameNum)])
        return ((ll - 1) * inc + frameLen / 2) / fs
    # 分帧函数
    def enframe(self,x, win, inc=None):
        nx = len(x)
        if isinstance(win, list) or isinstance(win, np.ndarray):
            nwin = len(win)
            nlen = nwin  # 帧长=窗长
        elif isinstance(win, int):
            nwin = 1
            nlen = win  # 设置为帧长
        if inc is None:
            inc = nlen
        nf = (nx - nlen + inc) // inc
        frameout = np.zeros((nf, nlen))
        indf = np.multiply(inc, np.array([i for i in range(nf)]))
        for i in range(nf):
            frameout[i, :] = x[indf[i]:indf[i] + nlen]
        if isinstance(win, list) or isinstance(win, np.ndarray):
            frameout = np.multiply(frameout, np.array(win))
        return frameout
    # 加窗
    def hanning_window(self,N):
        nn = [i for i in range(N)]
        return 0.5 * (1 - np.cos(np.multiply(nn, 2 * np.pi) / (N - 1)))
    # 短时傅里叶变换
    def STFFT(self,x, win, nfft, inc):
        xn = self.enframe(x, win, inc)
        xn = xn.T
        y = np.fft.fft(xn, nfft, axis=0)
        return y[:nfft // 2, :]

    #自己画语谱图的主函数
    def demain(self):

        data,fs=librosa.load(self.filename,sr=None,mono=False)      # sr=None声音保持原采样频率， mono=False声音保持原通道数
        y = self.STFFT(data, self.win, self.nfft, self.inc)

        FrequencyScale = [i * fs / self.wlen for i in range(self.wlen // 2)]  # 频率刻度
        frameTime = self.FrameTimeC(y.shape[1], self.wlen, self.inc, fs)  # 每帧对应的时间
        LogarithmicSpectrogramData = 10 * np.log10((np.abs(y) * np.abs(y)))  # 取对数后的数据

        plt.pcolormesh(frameTime, FrequencyScale, LogarithmicSpectrogramData)
        plt.colorbar()
        plt.ylabel('Frequency[Hz]')
        plt.xlabel('Time[s]')
        # plt.title('spec_real.wav',fontsize=12,color='black')
        plt.title('spec_target.wav',fontsize=12,color='black')
        # plt.savefig('G:\\pytorch\\vc-master\\convert-checkpoint\\net1\\MFCC\\100000_spec_real.png')
        plt.savefig('G:\\pytorch\\vc-master\\convert-checkpoint\\net1\\MFCC\\100000_spec_target.png')

        plt.show()
if __name__ == '__main__':
    #源说话人音频
    # filename = "G:\\pytorch\\vc-master\\convert-checkpoint\\net1\\ckpt_100000.pth-spec_real.wav"
    #目标说话人音频
    filename = "G:\\pytorch\\vc-master\\convert-checkpoint\\net1\\ckpt_100000.pth-spec_target.wav"
    wlen = 512
    inc = 256

    mySpec = MySpec(filename, wlen, inc)
    mySpec.demain()
#\myplot.png