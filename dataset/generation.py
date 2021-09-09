# import sys
# sys.path.append('F:\code\python\data_mining\possc2.0')
import re
import pandas as pd
from pathlib import Path
from constants.parameters import *

def generation():
    """
    将原始表转为stove1.csv和stove2.csv作为训练用数据
    """
    # 数据目录和目标目录
    source_path = Path.home() / 'SteelmakingData'
    target_path = Path.cwd() / 'dataset' / 'native'
    for sub_dir in source_path.iterdir():
        dir_name = sub_dir.name
        # 表格式转换
        df_list = table_trans(sub_dir)
        # 合并表
        df = table_merge(df_list)
        # 数据清洗，将df内所有数据转为数字类型
        df = cleanout(df)
        # 根据钢种类型对数据进行分类
        result = classify(df)
        # 保存数据
        result.to_csv(Path(target_path, dir_name + '.csv'), encoding='gbk', index=0)

def table_trans(path):
    """
    将原始数据表转为单属性英文属性表

    Parameters
    ----------
    path: pathlib.Path
        原始数据的文件夹名

    Return
    ------
    df_list: list
        转换后的DataFrame列表
    """
    df_list = []
    for fileName in path.iterdir():
        df = pd.read_csv(fileName, encoding='gbk')
        # 提取特定列
        df = df.loc[:, factors_zh]
        # 将列名转为英文
        df.columns = factors_en
        df_list.append(df)
    return df_list

def table_merge(df_list):
    """
    将list内的DataFrame数据进行合并

    Parameters
    ----------
    df_list: list
        需要合并的DataFrame列表

    Return：DataFrame
        合并后的DataFrame
    """
    res_data = pd.concat(df_list, axis=0)
    return res_data

def cleanout(df):
    """
    对DataFrame数据进行清洗，将非round数据转为round
    """
    # 对df中的时间类型数据进行转换
    for column in factors_time:
        df[column] = df[column].apply(transforTime)
    # 对df中的百分比类型数据进行转换
    for column in factors_percent:
        df[column] = df[column].apply(transforPercent)
    # 将df中所有数据转为数字类型
    for column in factors_symbol:
        df[column] = df[column].apply(pd.to_numeric, errors='coerce')
    return df

def classify(df):
    """
    根据钢种类型对数据进行划分
    """
    df = df.set_index(['steel_type'], drop=False)
    df = df.sort_index()
    return df

def transforTime(time):
    """
    时间转数字
    """
    if re.search('\d[:]\d', str(time)):
        lists = str(time).split(':',1)
        res = float(lists[0]) + float(lists[-1])/60
        return round(res, 2)
    else:
        return None

def transforPercent(obj):
    """
    百分数转数字
    """
    if re.search('\d[%]', str(obj)):
        lists = str(obj).split('%', 1)
        res = float(lists[0])
        return round(res, 2)
    else:
        return None

if __name__ == '__main__':
    generation()
