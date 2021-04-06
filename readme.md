# 钢冶炼数据处理与成分预测
This is a pytorch deep learning project that recognizes data processing and component prediction for steelmaking.

钢冶炼中生产数据处理与成分预测的Pytorch深度学习项目

<p align="center">
    <img src="docs/intro.gif" width="480">
</p>

## 安装

### 下载部分数据文件‘SteelmakingData’

冶炼数据转炉操作数据表下载：

转炉数据(部分）：[Nutstore 坚果云](https://www.jianguoyun.com/p/DckpewMQnqiiCRjfp-QD )

                [百度网盘](https://pan.baidu.com/s/13QVRQzEtev9LSaTTDxF2UA) 提取码：3ej1

放置在：
```
(用户文件夹)/SteelmakingData

# 用户文件夹 在 Windows下是'C:\Users\(用户名)'，在Linux下是 '/home/(用户名)'
```

### 安装Pytorch和其它依赖：
```bash
# Python 3.8.5
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install ujson
pip install visdom opencv-python imgaug scikit-learn joblib 
```

## 参数更改

使用时需要更改部分参数：
    1） possc.py 文件中第3，4行，将路径改为自己的路径
    2） constants 目录下的parameters.py文件保存了系统超参数，根据需要更改（建议根据需要更改self.epochs,默认4000)

## 使用

```bash

# 使用时首先应生成训练用总数据
python possc.py -g

# 以下部分均需要较复杂命令行参数，通常通过该系统java web部分进行调取
# %steelType 代表该位置的期望输入，可在parameters中进行查看
# 若要测试 请按被注释部分的操作进行测试

# 模型训练
python possc.py -t %steelType %stoveNum %input_factors %output_factors
# python ./train/train_bp_model.py

# 模型测试
python possc.py -e %steelType %stoveNum %code_16
# python ./pred/test_bp_model.py

# 模型预测
python possc.py -p %steelType %stoveNum %input_factorNum %code_16
# python ./pred/test_bp_model.py

# 线性回归预测
python possc.py -l %steelType %stoveNum %input_factors %output_factors %input_factorsNum
# python ./pred/pred_linear_regression.py

# 数据预处理
python possc.py -s %steelType %stoveNum %input_factors %output_factors
# python ./preprocess/data_preprocess.py

# 缓存清理（针对特定钢种根据编码部分清理）
python possc.py -n %steelType %stoveNum %code_16

# 缓存清理（针对特定钢种将所有数据清理）
python possc.py -c %steelType %stoveNum

# 还愿为初始状态（一键清空所有数据）
python possc.py -r

# 训练等其它功能见帮助
python ctpgr.py --help
```

## 前端页面java web部分
https://github.com/lanhas/possc-javaWeb

