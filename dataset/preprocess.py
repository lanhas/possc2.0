import sys
sys.path.append(r'F:\code\python\data_mining\possc2.0')
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from constants.parameters import *

def preprocess(steelType, stoveNum, input_factors, output_factors, paths):
    """
    数据预处理，根据条件从总数据中抽取满足该条件的数据并进行处理得到数据集

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
    paths: pathlib.Path
        路径，包含path_train, path_test, path_inputScaler, path_outputScaler
    
    Return
    ------
    None
    """
    # 将符合要求的数据提取并合并成一个df
    df = native(steelType, stoveNum, input_factors, output_factors)
    # 对该df进行数据清洗
    df = cleanout(df)
    # 构建特征工程
    df = Featurization(df)
    # 数据集划分
    df_input = df.loc[:, input_factors]
    df_output = df.loc[:, output_factors]
    split(df_input, df_output, paths)

def native(steelType, stoveNum, input_factors, output_factors):
    """
    数据预处理，根据条件从总数据中抽取满足该条件的数据作为数据集

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
    
    Return
    ------
    result: DataFrame
        提取的数据
    """
    # 读取文件
    fileName = 'stove' + str(stoveNum) + '.csv'
    df_path = Path.cwd() / 'dataset' / 'native' / fileName
    df = pd.read_csv(df_path, encoding='gbk')
    # 根据钢种类型进行分类
    df = df.set_index(['steel_type'], drop=False)
    df = df.loc[steelType]
    # 将输入和输出属性进行合并
    columns = input_factors + output_factors
    df_res = df.loc[:, columns]
    df_res.index = range(len(df_res))
    return df_res

def cleanout(df):
    """
    清洗掉DataFrame内的空值和异常值
    """
    # 去掉空值
    df = df.dropna(axis=0, how='any').reset_index()
    # 去掉异常值
    error_rows = error_index(df)
    df_res = df.drop(error_rows, axis=0).reset_index()
    return df_res

def Featurization(df):
    """
    特征工程，对df进行特征转换
    """
    return df

def split(df_input, df_output, paths):
    """
    将数据归一化、保存归一化模型并划分训练集和测试集

    Parameters
    ----------
    df_input: pd.DataFrame
        影响因素，可能对结果产生影响的输入因素
    df_output: pd.DataFrame
        回归因素，输出因素
    
    Return
    ------
    None
    """
    path_train, path_test, path_inputScaler, path_outputScaler = paths
    # 对数据进行归一化
    minmax_inputScaler = MinMaxScaler()
    minmax_outputScaler = MinMaxScaler()
    
    df_input = minmax_inputScaler.fit_transform(df_input)
    df_output = minmax_outputScaler.fit_transform(df_output)

    df_input = pd.DataFrame(df_input)
    df_output = pd.DataFrame(df_output)
    # 保存归一化模型
    joblib.dump(minmax_inputScaler, path_inputScaler)
    joblib.dump(minmax_outputScaler, path_outputScaler)

    df_norm = pd.concat((df_input, df_output), axis=1)
    # 训练集、测试集划分
    df_train, df_test = train_test_split(df_norm, test_size=test_size, random_state=42)

    # 保存文件
    df_train.to_csv(path_train, encoding='gbk', index=0)
    df_test.to_csv(path_test, encoding='gbk', index=0)

def error_index(df, coef=3.5):
    """
    查找异常值，返回异常的数据索引
    """
    error_rows = []
    for j in range(df.shape[1]):
        Percentile = np.percentile(df.iloc[:, j],[0,25,50,75,100])
        IQR = Percentile[3] - Percentile[1]
        UpLimit = Percentile[3]+IQR * coef
        DownLimit = Percentile[1]-IQR * coef
        for i in range(df.iloc[:, j].shape[0]):
            if df.iloc[i, j] > UpLimit or df.iloc[i, j] < DownLimit:
                error_rows.append(i)
    error_rows = np.unique(error_rows)
    return error_rows

if __name__ == '__main__':
    preprocess('Q235B-Z', 1, input_factorsTest, output_factorsTest)
