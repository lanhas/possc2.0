# 钢冶炼数据处理与成分预测2.0
This is a new pytorch deep learning project that recognizes data processing and component prediction for steelmaking.

钢冶炼中生产数据处理与成分预测的Pytorch深度学习项目2.0

# 本次升级内容：
① 去除了繁琐的类关系，使用函数代替，简化了代码；  

② 优化了所有代码的逻辑关系，缩减了代码量；  

③ 优化了网络结构，减少了训练量；  

④ 重写了编解码部分，修复了bug，优化了逻辑，扩展了json文件的内容；  

⑤ 增加了模型名的查找和更改、模型精度的查找等功能；  

⑥ 为所有代码加上了详细的注释。

<p align="center">
    <img src="docs/intro.gif" width="480">
</p>

## 安装

### 下载部分数据文件‘SteelmakingData’

冶炼数据转炉操作数据表下载：  

下载地址：  

[百度网盘](https://pan.baidu.com/s/1q6q3-damswB9u0kkO0N1uQ ) 提取码：pyzc  

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
pip install visdom scikit-learn joblib 
```

## 参数更改

使用时需要更改部分参数:  

① possc.py 文件中第3，4行，将路径改为自己的路径  

② constants 目录下的parameters.py文件保存了冶炼参数，根据需要更改

## 使用

```bash

# 使用时首先应生成训练用总数据
python possc.py -g

# 以下部分均需要较复杂命令行参数，通常通过该系统java web部分进行调取
# %steelType 代表该位置的期望输入，可在parameters中进行查看
# 若要测试 请按被注释部分的操作进行测试

# 模型训练
python possc.py -t %steelType %stoveNum %input_factors %output_factors
# python ./train/train_nn.py
"""
模型训练

Parameters
----------
steelType: str
    钢种类型
stoveNum: int
    炉号，{1, 2}optional
input_factors: list
    影响因素，可能对结果产生影响的输入因素
output_factors: list
    回归因素，输出因素
"""

# 模型测试
python possc.py -e %steelType %stoveNum %code_16
# python ./pred/pred_nn.py
"""
对训练结果进行测试

Parameters
----------
steelType: str
    钢种类型
stoveNum: int
    炉号
code_16: str
    模型的16进制编码
"""

# 模型预测
python possc.py -p %steelType %stoveNum %code_16 %input 
"""
成分预测

Parameters
----------
steelType: str
    钢种类型
stoveNum: int
    炉号
code_16: str
    模型的16进制编码
input: ndarray
    预测数据
"""

# 线性回归预测
python possc.py -l %steelType %stoveNum %input_factors %output_factors %input
# python ./pred/pred_lr.py
"""
线性回归
steelType: str
钢种类型
stoveNum: int
    炉号，{1, 2}optional
input_factors: list
    影响因素，可能对结果产生影响的输入因素
output_factors: list
    回归因素，输出因素
input: list(可选参数)
    输入数据, if None，只回归出关系式；else 根据input数值计算出结果
"""

# 数据预处理
python possc.py -s %steelType %stoveNum %input_factors %output_factors

# 缓存清理（针对特定钢种根据编码部分清理）
python possc.py -n %steelType %stoveNum %code_16

# 缓存清理（针对特定钢种将所有数据清理）
python possc.py -c %steelType %stoveNum

# 还愿为初始状态（一键清空所有数据）
python possc.py -r

# 修改模型名
python possc.py -a %steelType %stoveNum %code_16 %new_name
"""
通过模型编码更改模型名

Parameters
----------
steelType: str
    钢种类型
stoveNum: int
    炉号，{1, 2}optional
code_16: str
    需要修改名称的模型编码
new_name: str
    新名称
"""

# 查找
python possc.py -f %steelType %stoveNum %mode %value
"""
根据条件查找结果

Parameters
----------
steelType: str
    钢种类型
stoveNum: int
    炉号，{1, 2}optional
mode: str
    查找模式,{'k2n', 'n2k', 'k2a', 'n2a'}optional
    k2n: (key to name)根据模型编码查找模型名
    n2k: (name to key)根据模型名查找模型编码
    k2a: (key to accuracy)根据模型编码查找模型精度
    n2a: (name to accuracy)根据模型名查找模型精度
value: str
    查找值
"""

# 训练等其它功能见帮助
python ctpgr.py --help
```

## 前端页面java web部分

