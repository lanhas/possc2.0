import sys
sys.path.append(r'F:\code\python\data_mining\possc2.0')

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
    #--------------------------------#
    #-----------超参数----------------#
    #--------------------------------#
    enable_vis = True
    update = False
    test = True

    # train
    lr = 2e-4
    epoches = 150
    batch_size = 64
    val_batch_size = 64
    best_val = 0.2
    name = ['tarin_loss', 'val_loss', 'acc_test']

    # setup visualization
    vis = Visdom(port=10086) if enable_vis else None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: %s" %device)

    # setup dataloader
    code_16, train_loader, val_loader = getTrainloader(steelType, stoveNum, input_factors, output_factors, batch_size, val_batch_size, update=update)
    if test == True:
        test_input, test_output = getTestdata(steelType, stoveNum, input_factors, output_factors)
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
        if epoch % 10 == 0:
            print("Epoch:{}".format(epoch))
            print("train loss: {}".format(loss_trained))
            print("val loss: {}".format(loss_valed))
            print("acc_test:{}".format(acc_test))
        vis.line(np.column_stack((loss_trained, loss_valed, acc_test)), [epoch], win='train_log', update='append', opts=dict(title='losses', legend=name))
        if loss_valed < best_val:
            torch.save(model.state_dict(), model_path)
    print('训练完成，模型名为:%s' %(code_16))
    torch.save(model.state_dict(), model_path)

def accuracy(predict_list, original_list):
        y_pred = np.asarray(predict_list)
        y_true = np.asarray(original_list)
        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        return 1 - u/v
def plot_hisloss(loss_trained, loss_valed):
    plt.title('Loss')
    plt.plot(loss_trained, label='train_loss', color='deepskyblue')
    plt.plot(loss_valed, label='val_loss', color='pink')
    plt.legend(['train_loss', 'val_loss'])
    plt.show()


if __name__ == '__main__':
    train('Q235B-Z', 1, input_factorsTest, output_factorsTest)