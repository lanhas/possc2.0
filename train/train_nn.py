import sys
sys.path.append('.')
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
from dataset.dataloader import *

def train(steelType, stoveNum, input_factors, output_factors):
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
    #-------------------------------------------------------------------#
    #------------------------------ 超参数 ------------------------------#
    #-------------------------------------------------------------------#

    # 训练设置
    enable_vis = True     # 使用visdom可视化训练过程
    update = False           # 更新训练数据，if True 每次训练时都先生成一遍训练数据，else当本地数据存在时，直接加载训练
    simple_print = False    # 是否输出最简训练信息
    plot = True             # 绘制训练结果

    # 参数设置
    lr = 2e-4
    epochs = 120
    batch_size = 64
    split_size = 0.15       # 训练集、验证集比例
    name = ['loss_train','accu_train', 'loss_valid', 'accu_valid', 'accu_test']

    # 可视化设置
    vis = Visdom(port=10086) if enable_vis else None
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not simple_print:
        print("Device: %s" %device)

    # DATA
    code_16, train_loader, valid_loader = load_dataset(steelType, stoveNum, input_factors, output_factors, batch_size, split_size, update=update)
    dataloader = {'train': train_loader, 'valid': valid_loader}
    # setup model
    model = BpNet(len(input_factors), len(output_factors)).to(device)
    # setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 等间隔调整学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    losser = nn.MSELoss()
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
    # init
    loss_list = {'train': [], 'valid': []}
    accu_list = {'train': [], 'valid': [], 'test': []}

    #----------------------------------------train Loop----------------------------------#
    for e in range(epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model = set_training(model, True)
            else:
                model = set_training(model, False)
            run_loss = 0.0
            run_accu = 0.0
            # with tqdm(dataloader[phase], desc=phase) as iterator:

            for batch_x, batch_y in dataloader[phase]:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # Forward
                out = model(batch_x)
                loss = losser(out, batch_y)

                if phase == 'train':
                    # Backward
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                run_loss += loss.item()
                accu = accuracy(out.detach().cpu().numpy(),
                                batch_y.detach().cpu().numpy())
                run_accu += accu
                # break
            loss_list[phase].append(run_loss / len(dataloader[phase]))
            accu_list[phase].append(run_accu / len(dataloader[phase]))
            # break

        # evaluation
        model = set_training(model, False)
        _, x_test, y_test = load_testData(steelType, stoveNum, code_16)
        test_x = torch.as_tensor(x_test, dtype=torch.float32).to(device)
        output = model(test_x).detach().cpu().numpy()
        acc_test = accuracy(output, y_test)
        if acc_test < 0:
            acc_test = 0
        # 学习率调整
        scheduler.step()
        accu_list['test'].append(acc_test)

        currLoss_train = loss_list['train'][-1]
        currAccu_train = accu_list['train'][-1] if accu_list['train'][-1] > 0 else 0
        currLoss_valid = loss_list['valid'][-1]
        currAccu_valid = accu_list['valid'][-1] if accu_list['valid'][-1] > 0 else 0
        currAccu_test = accu_list['test'][-1] if accu_list['test'][-1] > 0 else 0
        if e % 10 == 0 and simple_print == False:
            print('Epoch{}/{}'.format(e, epochs - 1))
            print('Summary epoch:\n - Train loss: {:.4f}, accu: {:.4f}\n - Valid loss:'
                    ' {:.4f}, accu: {:.4f}\n, TestAcc: {}'.format(currLoss_train, currAccu_train,
                                                    currLoss_valid, currAccu_valid, currAccu_test))
        if enable_vis:
            vis.line(np.column_stack((currLoss_train, currAccu_train, currLoss_valid, currAccu_valid, currAccu_test)), \
                        [e], win='train_log', update='append', opts=dict(title='training', legend=name))
        # save
        # 大于accuracy的模型将被保存
        if currAccu_test > best_acc:
            best_acc = currAccu_test
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
        plot_result(loss_list['train'], loss_list['valid'], accu_list['test'])
    del train_loader, valid_loader

def set_training(model, mode=True):
    for p in model.parameters():
        p.requires_grad = mode
    model.train(mode)
    return model

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
    plt.plot(index, loss_trained, label='predict', color='deepskyblue')
    plt.plot(index, loss_valed,  label='original', color='orange')
    plt.plot(index, acc,  label='accuracy', color='pink')
    plt.legend(['train_loss', 'val_loss', 'accuracy'])
    plt.show()

if __name__ == '__main__':
    train('Q235B-Z', 1, input_factorsTest, output_factorsTest)