# -*- Coding:utf-8 -*-
# Author:Universe Feng

import torch
import librosa

# print(torch.cuda.device_count())

d=list("abcdefghijklmnopqrstuvwxyz")
s=input("请输入字母，不用输入空格:")
s=[(d.index(i.lower()),i) for i in s]
s.sort()
print("".join([i[1] for i in s]))
