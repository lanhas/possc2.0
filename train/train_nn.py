import sys
sys.path.append(r'F:\code\python\data_mining\possc2.0')
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from constants.parameters import *
from pathlib import Path
from visdom import Visdom
from models.bp_network import BpNet
from dataset.dataloader import getTrainloader, getTestdata

def train(steelType, stoveNum, input_factors, output_factors):
    #-------------------------------------------------------------------#
    #------------------------------ 超参数 ------------------------------#
    #-------------------------------------------------------------------#

    # 训练设置
    enable_vis = False      # 使用visdom可视化训练过程
    update = True           # 更新训练数据，if True 每次训练时都先生成一遍训练数据，else当本地数据存在时，直接加载训练
    simple_print = False    # 是否输出最简训练信息
    test = True             # 训练时使用测试集进行动态验证
    plot = True             # 绘制训练结果

    # 参数设置
    lr = 2e-4
    epoches = 150
    batch_size = 64
    val_batch_size = 64
    name = ['tarin_loss', 'val_loss', 'acc_test']

    # 可视化设置
    vis = Visdom(port=10086) if enable_vis else None
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not simple_print:
        print("Device: %s" %device)

    # setup dataloader
    code_16, train_loader, val_loader = getTrainloader(steelType, stoveNum, input_factors, output_factors, batch_size, val_batch_size, update=update)
    if test == True:
        _, test_input, test_output = getTestdata(steelType, stoveNum, code_16)
    # setup model
    model = BpNet(len(input_factors), len(output_factors)).to(device)
    # setup optimizer
    opt = optim.Adam(model.parameters(), lr=lr)
    # 等间隔调整学习率
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size = 30, gamma = 0.1, last_epoch=-1)
    loss = nn.MSELoss()
    # setup model path
    modelfolder_path = Path.cwd() / 'models' / 'models_trained' / steelType
    modelfolder_path.mkdir(parents=True, exist_ok=True)
    model_path = modelfolder_path / Path('stove' + str(stoveNum) + '#' + code_16 + '.pth')
    # 模型精确度获取
    coder_path = Path.cwd() / 'models' / 'models_coder' / steelType / Path('stove' + str(stoveNum) + '.json')
    with open(coder_path, 'r', encoding='gbk') as json_file:
        dict_coder = json.load(json_file)
        json_file.close()
    best_acc = dict_coder[code_16]['accuracy']      # 模型精确度，大于该精确度的模型将替换之前的模型
    # Restore
    loss_his_train = []
    loss_his_val = []
    acc_his = []

    #----------------------------------------train Loop----------------------------------#
    for epoch in range(epoches):
        loss_train = 0
        loss_val = 0
        model.train()
        for _, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = torch.as_tensor(batch_x, dtype=torch.float32).to(device)
            batch_y = torch.as_tensor(batch_y, dtype=torch.float32).to(device)
            outputs = model(batch_x)
            loss_step = torch.tensor(.0).to(device)
            for i in range(len(output_factors)):           
                loss_step += loss(outputs[:, i], batch_y[:, i])
            opt.zero_grad()
            loss_step.backward()
            opt.step()
            loss_train += float(loss_step.item())
        model.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = torch.as_tensor(batch_x, dtype=torch.float32).to(device)
                batch_y = torch.as_tensor(batch_y, dtype=torch.float32).to(device)
                outputs = model(batch_x)
                loss_step = torch.tensor(.0).to(device)
                for i in range(len(output_factors)):
                    loss_step += loss(outputs[:, i], batch_y[:, i])
                loss_val += float(loss_step.item())
        if test:
            test_x = torch.as_tensor(test_input, dtype=torch.float32).to(device)
            output = model(test_x).detach().cpu().numpy()
            acc_test = accuracy(output, test_output)
            if acc_test < 0:
                acc_test = 0
        # 学习率调整
        scheduler.step()
        # 计算loss
        loss_trained = loss_train / len(train_loader)
        loss_valed = loss_val / len(val_loader)
        
        loss_his_train.append(loss_trained)
        loss_his_val.append(loss_valed)
        acc_his.append(acc_test)
        # 精简模型将不输出训练细节
        if not simple_print:
            if epoch % 10 == 0:
                print("Epoch:{}".format(epoch))
                print("train loss: {}".format(loss_trained))
                print("val loss: {}".format(loss_valed))
                print("acc_test:{}".format(acc_test))
        if enable_vis:
            vis.line(np.column_stack((loss_trained, loss_valed, acc_test)), [epoch], win='train_log', update='append', opts=dict(title='losses', legend=name))
        # 大于accuracy的模型将被保存
        if acc_test > best_acc:
            best_acc = acc_test
            # 更改json文件中的最高精度
            with open(coder_path, 'w', encoding='gbk') as json_file:
                dict_coder[code_16]['accuracy'] = round(best_acc, 2)
                json_str = json.dumps(dict_coder)
                json_file.write(json_str)
                json_file.close()
            # 保存模型
            torch.save(model.state_dict(), model_path)

    print("训练完成\n模型编码: {}\n 模型名：{}\n 模型精度：{:.2f}\n 最优精度：{:.2f}\n".format \
         (code_16, code_16, acc_test*10+90, best_acc*10+90))
    if plot:
        plot_result(loss_his_train, loss_his_val, acc_his)

def accuracy(predict_list, original_list):
    y_pred = np.asarray(predict_list)
    y_true = np.asarray(original_list)
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - u/v

def plot_result(loss_trained, loss_valed, acc):
    plt.figure(figsize=(9, 6))
    index = range(len(loss_trained))
    plt.title('Train Result')
    plt.xticks(np.arange(0, len(index), step=5))
    plt.ylim(0, 1)
    plt.plot(index, loss_trained, label='predict', color='r')
    plt.plot(index, loss_valed,  label='original', color='pink')
    plt.plot(index, acc,  label='accuracy', color='orange')
    plt.legend(['train_loss', 'val_loss', 'accuracy'])
    plt.show()

if __name__ == '__main__':
    train('Q235B-Z', 1, input_factorsTest, output_factorsTest)