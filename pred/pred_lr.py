import sys
sys.path.append('F:\code\python\data_mining\possc2.0')

import numpy as np
from dataset.dataloader import getTraindata
from sklearn.linear_model import LinearRegression

def linearRegression(steelType, stoveNum, input_factors, output_factors, input):
    """
    线性回归
    """
    update = False      # 每次训练是否都得生成数据，if false，可以直接读取已有数据

    # 对输入数据进行解码
    input_values = []
    for i, v in enumerate(input):
        if v != '':
            input_values.append(float(v))
    # 将待预测样本转为二维np矩阵
    input_values = np.array(input_values).reshape(1, -1)
    # 构造拟合模型
    model = LinearRegression()
    # 获取训练数据
    df = getTraindata(steelType, stoveNum, input_factors, output_factors, update)
    data_input = np.array(df.loc[:, input_factors])
    data_output = np.array(df.loc[:, output_factors])
    # 数据拟合
    model.fit(data_input, data_output)
    intercept = model.intercept_
    coef = model.coef_
    # 数据预测
    result = model.predict(input_values)
    # 输出预测结果
    lr_print(input_factors, output_factors, intercept, coef)
    print('预测结果：{:.2f}'.format(*result[0]))

def lr_print(input_factors, output_factors, intercept, coef):
    coef_round2 = list(coef[0]).apply(lambda x: round(x, 2))
    print(coef_round2)
    res = output_factors[0] + '='
    for idx, val in enumerate(coef[0]):
        res = res + str(round(val, 2)) + '*' + input_factors[idx] + '+'
    res = res + str(round(intercept[0], 2))
    # print("最佳拟合线:截距{:.2f}\n回归系数：{}",intercept[0], coef[0])
    # print('-----------------------------------------------------------------------')
    print(res,'\n')

        
if __name__ == '__main__':
    from constants.parameters import *
    input_factorNum = [0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1]
    linearRegression('Q235B-Z', 1, input_factorsTest, output_factorsRegression, input_factorNum)
