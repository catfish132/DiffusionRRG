# DiffusionRRG

## Introduction
this repository is for Radiology Report Generation(RRG) with BitDiffusion. This is developed based on [SCD-Net](https://arxiv.org/abs/2212.03099).

## Acknowledgement
This code used resources from [X-Modaler Codebase](https://github.com/YehLi/xmodaler) and [bit-diffusion code](https://github.com/lucidrains/bit-diffusion). We thank the authors for open-sourcing their awesome projects.

## License

MIT

## 开发日志
### 为什么生成的句子长度不够？
1. 采用非自回归的方式，自回归是有强烈的因果关系的，每一个单词都根据前面的单词来判断这个时间步骤是否应该生成EOS。但是diffusion没有因果关系。
2. 预测eos 之后的单词仍然会参与到loss计算中，但是在推理时被去除了。这导致loss和bleu指标脱节。
3. 无法使用beam search