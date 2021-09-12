import sys
sys.path.append('.')
import numpy as np
from dataset.dataloader import getTraindata
from sklearn.linear_model import LinearRegression

def linearRegression(steelType, stoveNum, input_factors, output_factors, input=None):
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
    update = False      # 每次训练是否都得生成数据，if false，可以直接读取已有数据

    # 对输入数据进行解码
    if not input is None:
        input_values = []
        for i, v in enumerate(input):
            if v != '':
                input_values.append(float(v))
        # 将待预测样本转为二维np矩阵
        input_values = np.array(input_values).reshape(1, -1)
    # 构造拟合模型
    model = LinearRegression()
    # 获取训练数据
    _, df = getTraindata(steelType, stoveNum, input_factors, output_factors, update)
    data_input = np.array(df.loc[:, input_factors])
    data_output = np.array(df.loc[:, output_factors])
    # 数据拟合
    model.fit(data_input, data_output)
    intercept = model.intercept_
    coef = model.coef_
    # 输出预测结果
    lr_print(input_factors, output_factors, intercept, coef)
    # 数据预测
    if not input is None:
        result = model.predict(input_values)
        print('预测结果：{:.2f}'.format(*result[0]))

def lr_print(input_factors, output_factors, intercept, coef):
    coef_round2 = list(map(lambda x: round(x, 2), list(coef[0])))
    res = output_factors[0] + '='
    for idx, val in enumerate(coef_round2):
        res = res + str(val) + '*' + input_factors[idx] + '+'
    res = res + str(round(intercept[0], 2))
    print("最佳拟合线:\n截距{:.2f}\n回归系数：{}".format(intercept[0], coef_round2))
    print('-----------------------------------------------------------------------')
    print(res,'\n')

        
if __name__ == '__main__':
    from constants.parameters import *
    input_factorNum = [0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1]
    linearRegression('Q235B-Z', 1, input_factorsTest, output_factorsRegression, input_factorNum)
