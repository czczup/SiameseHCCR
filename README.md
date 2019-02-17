# Siamese-HCCR

## 1. 项目简介与依赖

本项目基于TensorFlow1.8与Python3.6进行开发，依赖库如下：

```
pillow
cv2
pandas
```

## 2. 项目运行

运行本项目请以此执行以下步骤：

### 2.1 数据集部署

- 将CASIA-Competition数据集中的所有gnt文件放置于`database/competition/`
- 将CASIA-HWDB1.0数据集中的所有gnt文件放置于`database/HWDB1.0`
- 将CASIA-HWDB1.1数据集中的所有gnt文件放置于`database/HWDB1.1`
- 运行如下代码

```
python generate_test_data.py  #测试集将gnt转为png，存放于database/test
python generate_train_data.py  #训练集集将gnt转为png，存放于database/train
```

### 2.2 开始训练

- 运行如下命令，自动完成tfrecord生成、模型训练、测试、数据集重构

```python
python main.py
```

### 2.3 查看训练日志

- 运行如下命令

```sh
tensorboard --logdir=file/logs
```

## 3. 项目结构树
```
│  generate_test_data.py  # 生成测试数据，用于将gnt转为png，只执行1次
│  generate_train_data.py  # 生成训练数据，用于将gnt转为png，只执行1次
│  generate_test_tfrecord.py  # 生成测试集tfrecord，目前没有用
│  generate_train_tfrecord.py  # 生成训练集tfrecord，对所有汉字随机采样
│  get_matching_template.py  # 计算汉字的特征向量，目前没有用
│  main.py  # 模型训练入口
│  model.py  # 模型定义在这里
│  reader.py  # 用于读取gnt文件并转为png
│  reconstruct_train_tfrecord.py  # 依据错误的识别结果重构数据集
│  test.py  # 测试
│  train.py  # 训练
│
└─file # 存放训练生成的文件
│   └─results
│   │   ├─train  # 存放训练集准确率文件
│   │   ├─test  # 存放测试集准确率文件
│   │   └─log  # 存放训练日志，只是空文件，标志训练进行的阶段，用于断点续训
│   ├─tfrecord  # 存放生成的tfrecord
│   ├─logs  # 存放tensorboard训练日志
│   └─models  # 存放保存的ckpt模型
│
└─database  # 存放训练数据
​     ├─competition  # 存放CASIA-Competition数据集的gnt文件
​     ├─HWDB1.0  # 存放CASIA-HWDB1.0数据集的gnt文件
​     ├─HWDB1.1  # 存放CASIA-HWDB1.1数据集的gnt文件
​     ├─test  # 存放由CASIA-Competition数据集生成的png文件
​     ├─train  # 存放由CASIA-HWDB1.0-1.1数据集生成的png文件
​     │  gb2312_level1.csv  # 对gb2312-80的一级常用汉字进行编号
​     |   count.csv  # 记录训练集中每一类的样本数
```