# Tacotron-2

Tacotron-2 模型的 PyTorch 实现，提出 Tacotron-2 的论文 [Natural TTS Synthesis By Conditioning Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf)。（持续完善ING）

## 目录结构

```
.
|--- audios/
|--- datasets/    # 数据集相关
     |--- audio/
     |--- text/
|--- helpers/     # 辅助类
|--- models/      # 模型相关
     |--- layers.py
     |--- losses.py
     |--- optimizers.py
     |--- tacotron.py
|--- tests/       # 测试代码
|--- utils/       # 一些通用方法
|--- .gitignore
|--- LICENSE
|--- README.md    # 说明文档（本文档）
|--- requirements.txt  # 依赖文件
|--- train.py       # 训练脚本
|--- synthesize.py  # 合成脚本
```

## 数据集

- [BZNSYP Dataset](https://www.data-baker.com/open_source.html)

## 快速开始

**步骤（1）**：克隆仓库

```shell
$ git clone https://github.com/atomicoo/Tacotron2-PyTorch.git
```

**步骤（2）**：安装依赖

```shell
$ conda create -n Tacotron2 python=3.7.9
$ conda activate Tacotron2
$ pip install -r requirements.txt
```

**步骤（3）**：合成语音

```shell
$ python synthesize.py
```

## 如何训练

