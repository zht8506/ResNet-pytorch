# Resnet

## 说明

本代码来源于 https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test5_resnet

本代码在此基础上写了大量注释，更加便于学习

论文：https://arxiv.org/abs/1512.03385

## 文件结构：

```
  ├── model.py: ResNet模型搭建
  ├── train.py: 训练脚本
  ├── predict.py: 单张图像预测脚本
  └── split_data.py：花分类数据集划分训练验证脚本
  └── batch_predict.py: 批量图像预测脚本
```

## 使用方法：

根据需要的resnet模型，下载预训练权重，然后运行train.py脚本。

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

## 训练数据：

本模型使用的是花分类数据集，如果要使用其他分类数据集，需要重写dataset模块。花分类数据集下载链接：

http://download.tensorflow.org/example_images/flower_photos.tgz

## 相关博文：

我的知乎关于resnet的论文笔记

> ResNet精读-残差连接使网络大大加深：

https://zhuanlan.zhihu.com/p/478096991

> 一个相关的笔记，里面介绍了花分类数据集的使用方法：

https://blog.csdn.net/qq_62932195/article/details/122094111

