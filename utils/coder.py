import sys
sys.path.append('.')
import math
import json
import numpy as np
from pathlib import Path
from constants.parameters import *

def encoder(steelType, stoveNum, input_factors, output_factors):
    """
    编码函数，使用一个本地json文件保存编码字典，其key为两位16进制码，value为若干位16进制码，
    该value为encoder16函数返回的16进制编码，该key为最后模型名所跟的两位16进制码

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
    code_16: str
        该数据对应的16进制编码
    """
    # 生成编码文件的存放目录
    coder_dir = Path.cwd() / 'models' / 'models_coder' / steelType
    coder_dir.mkdir(parents=True, exist_ok=True)
    coder_path = coder_dir / Path('stove' + str(stoveNum) + '.json')

    # 判断文件是否存在，若不存在则新建json文件
    if not coder_path.exists():
        with open(coder_path, 'w') as json_newfile:
            json.dump({}, json_newfile)
            dict_coder = {}
        json_newfile.close()
    else:
        with open(coder_path, 'r+') as json_file:
            dict_coder = json.load(json_file)
        json_file.close()

    # 获取编码key
    key = hex(len(dict_coder.keys()))

    # 获取编码Value
    code_value = encode_16(input_factors, output_factors)
    value = {"code_value": code_value,
             "name": key,
             "accuracy": 0}
    addition = True
    # 判断编码value是否在其中
    for i in range(len(dict_coder.keys())):
        if code_value == list(dict_coder.values())[i]['code_value']:
            addition = False
            break
    # 如果字典中没有该编码，则写入
    if addition:
        dict_coder[key] = value
        json_str = json.dumps(dict_coder)
        with open(coder_path, 'w', encoding='gbk') as json_file:
            json_file.write(json_str)
            json_file.close()
    else:
        return hex(len(dict_coder.keys()) - 1)
    return key

def encode_16(input_factors, output_factors):
    """
    将输入输出参数编码为16进制，编码方式为改进的onehot编码，先用若干位的二进制标记是否
    含有该参数，"1"表示有，随后通过每四位二进制合并为一个16进制的方式获得16进制编码

    Parameters
    ----------
    input_factors: list
        影响因素，可能对结果产生影响的输入因素
    output_factors: list
        回归因素，输出因素

    Return
    ------
    code_16: str
        16进制编码
    """
    # 初始化
    code_16 = []
    indexs_binX = np.zeros(len(input_factorsAll)).astype(int)
    indexs_binY = np.zeros(len(output_factorsAll)).astype(int)
    # 根据总属性列表，将勾选的属性位置为1
    for column in input_factors:
        indexs_binX[np.array(input_factorsAll)==column] = 1
    for column in output_factors:
        indexs_binY[np.array(output_factorsAll)==column] = 1
    # 二进制转16进制
    for i in range(math.ceil(len(input_factorsAll) / 4)):
        temp_10 = 8 * indexs_binX[4 * i] + 4 * indexs_binX[4 * i + 1] + 2 * indexs_binX[4 * i + 2] + indexs_binX[4 * i + 3]
        temp_16 = hex(temp_10)
        code_16.append(temp_16)
    for i in range(math.ceil(len(output_factorsAll)/ 4)):
        temp_10 = 8 * indexs_binY[4 * i] + 4 * indexs_binY[4 * i + 1] + 2 * indexs_binY[4 * i + 2] + indexs_binY[4 * i + 3]
        temp_16 = hex(temp_10)
        code_16.append(temp_16)
    return code_16

def decoder(steelType, stoveNum, code_16):
    """
    解码函数，用于将模型后所带的16进制码解码为输入输出参数

    Parameters
    ----------
    steelType: str
        钢种类型
    stoveNum: int
        炉号，{1, 2}optional
    code_16: str
        16进制编码

    Return
    ------
    input_factors: list
        影响因素，可能对结果产生影响的输入因素
    output_factors: list
        回归因素，输出因素

    """
    # 打开json文件中的编码字典
    coder_path = Path.cwd() / 'models' / 'models_coder' / steelType / Path('stove' + str(stoveNum) + '.json')
    with open(coder_path, 'r', encoding='gbk') as json_file:
        dict_coder = json.load(json_file)
        json_file.close()
    # 查找该编码是否存在
    if not code_16 in dict_coder.keys():
        raise ValueError('该编码不存在，请检查后重试!')
    # 取出该编码对应的value列表
    values = dict_coder[code_16]['code_value']
    # 解码
    input_factors, output_factors = decode_16(values)
    return input_factors, output_factors

def decode_16(values, num_input=10):
    """
    16进制解码函数，将一个若干16进制编码解码为输入和输出参数

    Parameters
    ----------
    values: list
        16进制编码列表
        example: ['0xf', '0xd', '0xe', '0xb', '0x7', '0xd', '0x1', '0x0', '0x0', '0x0', '0xf', '0x0', '0x0']
        前十位代表输入，后三位代表输出
    num_input: int
        影响因素所占的位数，默认为10
    num_output: int
        目标因素所占的位数，默认为3

    Return
    ------
    res_input: list
        影响因素，可能对结果产生影响的输入因素
    res_output: list
        回归因素，输出因素
    """
    # 初始化
    codeList_input = []     # 保存解码后的二进制字符
    codeList_output = []    # 保存解码后的二进制字符
    # index列表用于存放与codeList对应的属性信息
    res_input = []       # 保存影响因素的解码结果
    res_output = []      # 保存目标因素的解码结果

    # 对“影响因素”进行解码
    for val in values[:num_input]:
        str_bin4 = hex2bin4(val)
        for c in str_bin4:
            codeList_input.append(c)
    # 将二进制字符内“1”位置对应的input_factorsAll内的因素添加到编码结果中
    for i, val in enumerate(codeList_input):
        if val == '1':
            res_input.append(input_factorsAll[i])

    # 对“目标因素”进行解码
    for val in values[num_input:]:
        str_bin4 = hex2bin4(val)
        for c in str_bin4:
            codeList_output.append(c)
    # 将二进制字符内“1”位置对应的output_factorsAll内的因素添加到编码结果中
    for i, val in enumerate(codeList_output):
        if val == '1':
            res_output.append(output_factorsAll[i])
    return res_input, res_output

def hex2bin4(str_hex):
    """
    16进制字符串转4位2进制字符串，用于解码
    
    Parameters
    ----------
    str_hex: str
        16进制字符串
    
    Return
    ------
    str_bin: str
        4位2进制字符串
    """
    str_bin = bin(int(str_hex, 16))
    l = (len(str_bin) - 2) % 4
    if l > 0:
        str_bin = str_bin[:2] + ('0'*(4-l)) + str_bin[2:]
    return str_bin[2:]

def find(steelType, stoveNum, mode, value):
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
    """
    # 打开json文件中的编码字典
    coder_path = Path.cwd() / 'models' / 'models_coder' / steelType / Path('stove' + str(stoveNum) + '.json')
    with open(coder_path, 'r', encoding='gbk') as json_file:
        dict_coder = json.load(json_file)
        json_file.close()
    # 根据模型编码查找模型名
    if mode == 'k2n':
        if value in dict_coder.keys():
            name = dict_coder[str(value)]['name']
        else:
            name = None
        result = name
    # 根据模型名查找模型编码
    elif mode == 'n2k':
        for _, val in enumerate(dict_coder.keys()):
            if dict_coder[val]['name'] == value:
                key = val
            else:
                key = None
        result = key
    # 根据模型编码查找模型精度
    elif mode == 'k2a':
        if value in dict_coder.keys():
            acc = dict_coder[str(value)]['accuracy']
        else:
            acc = None
        result = acc
    # 根据模型名查找模型精度
    elif mode == 'n2a':
        for _, val in enumerate(dict_coder.keys()):
            if dict_coder[val]['name'] == value:
                acc = dict_coder[val]['accuracy']
            else:
                acc = None
        result = acc
    else:
        print('请输入正确的模式！')
        result = None
    print(result)
    return result

def rename(steelType, stoveNum, code_16, new_name):
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
    # 打开json文件中的编码字典
    coder_path = Path.cwd() / 'models' / 'models_coder' / steelType / Path('stove' + str(stoveNum) + '.json')
    with open(coder_path, 'r', encoding='gbk') as json_file:
        dict_coder = json.load(json_file)
        json_file.close()
    # 判断code_16是否符合
    if not code_16 in dict_coder.keys():
        print('该编码不存在，请检查后重试')
        return
    # 判断新名称是否与原名称相同
    if new_name == dict_coder[str(code_16)]['name']:
        print('新名称与原名称相同，请选择其他名称！')
    else:
        # 判断是否已有该名
        for _, val in enumerate(dict_coder.keys()):
            if dict_coder[val]['name'] == new_name:
                print('该命名已存在，请选择其他名称！')
                return
        # 修改模型名
        with open(coder_path, 'w', encoding='gbk') as json_file:
            dict_coder[str(code_16)]['name'] = new_name
            json_str = json.dumps(dict_coder)
            json_file.write(json_str)
            json_file.close()
            print("模型名修改成功！\n 模型编码：{}\n 模型名：{}".format(code_16, new_name))

if __name__ == '__main__':
    rename('Q235B-Z', 1, '0x1', 'test3')
