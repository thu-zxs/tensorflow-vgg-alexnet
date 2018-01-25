# 说明文档

## Guide

Tensorflow based VGG16 and AlexNet training, validation and test pipeline.

A batch generator with a shuffle in each epoch.

Friendly command line interface and log management.

With Kaggle Digit Recognition as Example.

Just try with ease.

## 运行环境

默认调用GPU，要求`train.csv`与`test.csv`与python文件处于同一目录.

## 训练

默认按train.csv样本顺序取前80%作为验证集

### AlexNet

```bash
python alexnet_train.py --lr 0.0001 --weightDecay 0.1 --maxIter 20000 --batchSize 30 --trainRatio 0.8
```

### VGG16

```bash
python vgg16_train.py --lr 0.0001 --weightDecay 0.1 --maxIter 20000 --batchSize 30
```


## 验证

默认按train.csv样本顺序取后20%作为验证集

### AlexNet

```bash
python alexnet_train.py --isVal --modelPath /path/to/your/model.cpkt
```

### VGG16

```bash
python vgg16_train.py --isVal --modelPath /path/to/your/model.cpkt
```

## 测试

将在当前目录生成submission.csv文件

### AlexNet

```bash
python alexnet_train.py --isTest --modelPath /path/to/your/model.cpkt
```

### VGG16

```bash
python vgg16_train.py --isTest --modelPath /path/to/your/model.cpkt
```

## more

```bash
python alexnet_train.py --help
python vgg16_train.py --help
```
