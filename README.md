# Feng-22级1班
# Research On Speaker Timber Conversion Method Based On PPG 

模型体系结构

这项工作的主要意义在于，我们可以生成目标说话人的话语，而不需要像<源的wav，目标的wav>， <wav，文本>或<wav，电话>这样的并行数据，而只需要目标说话人的波形。
(制作这些并行数据集需要付出很多努力。)
在这个项目中，我们所需要的只是目标说话者话语的一些波形，以及来自许多匿名说话者的一小组<wav, phone>对。

模型体系结构由两个模块组成:

1. Net1(音素分类)在每个时间步将某人的话语划分为一个音素类。
*音素与说话人无关，而波形与说话人有关。
2. Net2(语音合成)从电话中合成目标说话者的语音。

Net1是一个分类器。

*流程:wav ->谱图-> mfccs ->音素分布。
* Net1将声谱图分类为音素，每个时间步长包含60个英语音素。
*对于每个时间步，输入为对数幅度谱图，目标为音素距离。
*目标函数为交叉熵损失。
*使用了mitt。
*包含630个说话人的话语和相应的电话，说类似的句子。
*测试精度超过70%

Net2是一个合成器。

Net2包含一个子网Net1。

*流程:net1(wav ->谱图-> mfccs ->音素dist.) ->谱图-> wav
* Net2合成目标说话者的演讲。
输入/目标是一组目标说话人的话语。
*由于Net1已经在上一步进行了培训，剩下的部分只需要在这一步进行培训。
*损耗是输入和目标之间的重构误差。(L2距离)
*数据集
*目标(匿名女性):北极(公众)
*从谱图中还原波时的Griffin-Lim重构。

# #实现

# # #要求

* python 3.7
* pytorch == 1.5
* librosa == 0.7.2

# # #设置

采样率:16000hz
*窗口长度:25ms
*跳长:5ms

# # #过程

*训练阶段:Net1和Net2依次进行训练。
* Train1(training Net1)
*运行' train_net1.py '进行训练，运行' test_net1.py '进行测试。
*培训2(Net2培训)
*运行' train_net2.py '进行训练，运行' test_net2.py '进行测试。
培训1完成后再培训2 !
*转换阶段:前馈到Net2
*运行' convert.py '获取结果示例。
*检查Tensorboard的音频选项卡来收听样本。
*看一下Tensorboard的图像选项卡上的音素分布可视化。
* x轴表示音素类，y轴表示时间步长
第一类x轴表示沉默。
