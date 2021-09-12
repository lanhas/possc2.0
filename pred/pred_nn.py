import sys
sys.path.append('.')
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from models.bp_network import BpNet
from constants.parameters import *
from utils.coder import *
from dataset.dataloader import  getTestdata
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

def predict(steelType, stoveNum, code_16, input):
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
    # 对模型编码进行解码，得到输入输出属性
    input_factors, output_factors = decoder(steelType, stoveNum, code_16)
    # 加载模型
    model_path = Path.cwd() / 'models' / 'models_trained' / steelType / ('stove' + str(stoveNum) + '#' + code_16 + '.pth')
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = BpNet(len(input_factors), len(output_factors))
    model.load_state_dict(checkpoint)
    
    # 加载归一化模型，将预测结果反归一化得到最终结果
    path_outputScaler = Path.cwd() / 'models' / 'models_scaler' / steelType / ('stove' + str(stoveNum) + '#' + str(code_16) + '_output.pkl')
    outputScaler = joblib.load(path_outputScaler)
    
    # 转变输入数据
    test_x = torch.as_tensor(input, dtype=torch.float32).unsqueeze(0)
    # 预测
    with torch.no_grad():
        model = model.eval()
        output = model(test_x)
    predicts = outputScaler.inverse_transform(output).squeeze(0)
    print(predicts)
    for idx, val in enumerate(predicts):
        print("{}预测结果：{:.2f}\n".format(output_factors[idx], val))

def test(steelType, stoveNum, code_16):
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
    # 对模型编码进行解码，得到输入输出属性
    input_factors, output_factors = decoder(steelType, stoveNum, code_16)

    # 得到测试数据
    _, test_x, test_y = getTestdata(steelType, stoveNum, code_16)

    # 加载模型
    model_path = Path.cwd() / 'models' / 'models_trained' / steelType / ('stove' + str(stoveNum) + '#' + code_16 + '.pth')
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = BpNet(len(input_factors), len(output_factors))
    model.load_state_dict(checkpoint)
    
    # 加载归一化模型，将预测结果反归一化得到最终结果
    path_outputScaler = Path.cwd() / 'models' / 'models_scaler' / steelType / ('stove' + str(stoveNum) + '#' + str(code_16) + '_output.pkl')
    outputScaler = joblib.load(path_outputScaler)
    
    # 预测
    with torch.no_grad():
        model = model.eval()
        test_x = torch.as_tensor(test_x, dtype=torch.float32)
        output = model(test_x)
    predicts = outputScaler.inverse_transform(output)
    labels = outputScaler.inverse_transform(test_y)
    plot_result(predicts, labels, output_factors)

def accuracy(predict_list, original_list):
    """
    模型精度计算

    Parameters
    ----------
    predict_list: list
        预测结果
    original_list: list
        标签

    Return
    ------
    accuracy: float
        精确度
    """
    y_pred = np.asarray(predict_list)
    y_true = np.asarray(original_list)
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    accuracy = 1 - u/v
    return accuracy

def plot_result(predicts, labels, output_factors):
    """
    绘制结果
    """
    plot_pos = [221, 222, 223, 224]
    predicts = pd.DataFrame(predicts)
    labels = pd.DataFrame(labels)
    fig = plt.figure(figsize=(9, 6))
    index = range(len(predicts))
    for i, val in enumerate(output_factors):
        fig.add_subplot(plot_pos[i])
        plt.title('{}成分预测'.format(val))
        plt.plot(index, predicts.iloc[:, i], label='predict', color='deepskyblue')
        plt.plot(index, labels.iloc[:, i],  label='original', color='pink')
        acc = accuracy(predicts.iloc[:, i], labels.iloc[:, i])
        print('{}成分预测'.format(val))
        print('准确度{:.2f}'.format(acc))
        plt.legend(['predict', 'original'])
    plt.show()
        
if __name__ == '__main__':
    # 预测
    # input_factorNum = [0.1, 0.1, 0.1,0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1,0.1, 0.1]
    # bptest = predict('Q235B-Z', 1, '0x0', input_factorNum)
    # 测试
    test('Q235B-Z', 1, '0x0')
