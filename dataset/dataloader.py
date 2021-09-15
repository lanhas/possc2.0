import sys
sys.path.append('.')
from constants.parameters import *
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from utils.coder import *
from dataset.preprocess import preprocess
from torch.utils import data
from sklearn.model_selection import train_test_split

def load_dataset(steelType, stoveNum, input_factors, output_factors, batch_size, split_size=0.15, update=False):
    """
    获取训练数据，并将其封装为DataLoader对象

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
    batch_size: int
        训练集batch_size
    split_size: float
        训练集验证集的分割比例
    update: bool
        是否更新数据，if True 将根据训练参数重新更新一次训练用数据
    
    Return
    ------
    code_16: str
        此训练数据对应的编码
    train_loader: DataLoader
        训练集
    valid_loader: DataLoader
        验证集
    """
    # 获取训练data
    code_16, df = load_trainData(steelType, stoveNum, input_factors, output_factors, update)
    # 划分数据集
    if split_size > 0:
        x_train, x_valid, y_train, y_valid = train_test_split(np.array(df.loc[:, input_factors]), \
                                            np.array(df.loc[:, output_factors]), test_size=split_size, random_state=42)
        train_set = data.TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                       torch.tensor(y_train, dtype=torch.float32))
        valid_set = data.TensorDataset(torch.tensor(x_valid, dtype=torch.float32),
                                       torch.tensor(y_valid, dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_set, batch_size)
    else:
        train_set = data.TensorDataset(torch.tensor(df.loc[input_factors], dtype=torch.float32),
                                       torch.tensor(df.loc[output_factors], dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = None
    return code_16, train_loader, valid_loader


def load_trainData(steelType, stoveNum, input_factors, output_factors, update):
    """
    获取训练数据，返回其对应的dataframe

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
    batch_size: int
        训练集batch_size
    val_batch_size: int
        验证集batch_size
    update: bool
        是否更新数据，if True 将根据训练参数重新更新一次训练用数据
    
    Return
    ------
    code_16: str
        此训练数据对应的编码
    df_res: DataFrame
        训练数据
    """
    # 构建文件夹
    train_dir = Path.cwd() / 'dataset' / 'train' / steelType
    test_dir = Path.cwd() / 'dataset' / 'test' / steelType
    scaler_dir = Path.cwd() / 'models' / 'models_scaler' / steelType

    train_dir.mkdir(parents = True, exist_ok = True)
    test_dir.mkdir(parents = True, exist_ok = True)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    # 获得编码
    code_16 = encoder(steelType, stoveNum, input_factors, output_factors)

    # 获得文件的保存路径
    path_train = train_dir / Path('stove' + str(stoveNum) + '#' + str(code_16) + '.csv')
    path_test = test_dir / Path('stove' + str(stoveNum) + '#' + str(code_16) + '.csv')
    path_inputScaler = scaler_dir / Path('stove' + str(stoveNum) + '#' + str(code_16) + '_input.pkl')
    path_outputScaler = scaler_dir / Path('stove' + str(stoveNum) + '#' + str(code_16) + '_output.pkl')
    paths = (path_train, path_test, path_inputScaler, path_outputScaler)
    # 判断是否可以直接读取数据
    if update==True or not path_train.exists():
        preprocess(steelType, stoveNum, input_factors, output_factors, paths)
    df_res = pd.read_csv(path_train, encoding='gbk')
    return code_16, df_res


def load_testData(steelType, stoveNum, code_16):
    # 获得编码
    input_factors, output_factors = decoder(steelType, stoveNum, code_16)
    # 获得文件的保存路径
    path_test = Path.cwd() / 'dataset' / 'test' / steelType / ('stove' + str(stoveNum) + '#' + str(code_16) + '.csv')
    # 判断是否存在
    if not path_test.exists():
        raise ValueError('测试文件不存在，请点击训练进行生成！')
    df = pd.read_csv(path_test, encoding='gbk')
    df_input = df.loc[:, input_factors]
    df_output = df.loc[:, output_factors]
    # 训练集
    return code_16, np.array(df_input), np.array(df_output)

# class SteelmakingData(Dataset):
#     """
#     将数据封装为Dataset对象
#     """
#     def __init__(self, df, len_input):
#         self.df = df
#         self.len_input = len_input

#     def __len__(self):
#         return self.df.shape[0]

#     def __getitem__(self, index):
#         sample = (np.array(self.df.iloc[index, :self.len_input]),
#                  np.array(self.df.iloc[index, self.len_input:]))
#         return sample
