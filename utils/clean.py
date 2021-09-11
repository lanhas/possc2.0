from pathlib import Path

def rm_steelTypeAllFile(steelType):
    """
    清空指定钢种的所有数据
    """
    try:
        folder_clean(Path.cwd() / 'dataset' / 'train' / steelType)      # 清空训练集数据
        folder_clean(Path.cwd() / 'dataset' / 'test' / steelType)       # 清空测试集数据
        folder_clean(Path.cwd() / 'models' / 'models_coder' / steelType)    # 清空归一化模型
        folder_clean(Path.cwd() / 'models' / 'models_trained' / steelType)  # 清空训练模型
        print('钢种类型%s下所有数据删除成功！' %(steelType))
    except Exception  as e:
        print('钢种类型%s下所有数据删除失败！' %(steelType))
        print(type(e), e)
        
def rm_dataFile(steelType, stoveNum, code_16):
    """
    清除训练数据
    """
    fileName = Path('stove' + str(stoveNum) + '#' + code_16 + '.csv')
    path_trainFile= Path.cwd() / 'dataset' / 'train' / steelType / fileName
    path_testFile= Path.cwd() / 'dataset' / 'test' / steelType / fileName
    try:
        path_trainFile.unlink()
        print('训练数据文件%s删除成功！' %(str(fileName)))
    except Exception  as e:
        print('训练数据文件%s删除失败，请排查！' %(str(fileName)))
        print(type(e),e)

        try:
            path_testFile.unlink()
            print('测试数据文件%s删除成功！' %(str(fileName)))
        except Exception  as e:
            print('测试数据文件%s删除失败，请排查！' %(str(fileName)))
            print(type(e),e)

def rm_coderModel(steelType, stoveNum):
    """
    清除归一化模型
    """
    fileName =  Path('stove' + str(stoveNum) +'.json')
    path_modelsCoder = Path.cwd() / 'models' / 'models_coder' / steelType / fileName
    try:
        path_modelsCoder.unlink()
        print('编码器文件%s删除成功！' %(str(fileName)))
    except Exception  as e:
        print('编码器文件%s删除失败，请排查！' %(str(fileName)))
        print(type(e),e)
    
def rm_trainedModel(steelType, stoveNum, code_16):
    """
    清除训练好的模型
    """
    fileName = Path('stove' + str(stoveNum) + '#' + code_16 + '.pth')
    path_modelTrained = Path.cwd() / 'models' / 'models_trained' / steelType / fileName
    try:
        path_modelTrained.unlink()
        print('模型文件%s删除成功！' %(str(fileName)))
    except Exception  as e:
        print('模型文件%s删除失败，请排查！' %(str(fileName)))
        print(type(e),e)

def sysReset():
    """
    重置系统
    """
    try:
        folder_clean(Path.cwd() / 'dataset' / 'train')  # 清空训练集数据
        folder_clean(Path.cwd() / 'dataset' / 'test')   # 清空测试集数据
        folder_clean(Path.cwd() / 'models' / 'models_coder')    # 清空归一化模型
        folder_clean(Path.cwd() / 'models' / 'models_trained')  # 清空训练模型
        print('系统重置成功')
    except Exception as e:
        print('系统重置失败！请排查！')
        print(type(e), e)

def folder_clean(path_dir):
    """
    清空文件夹下的所有数据
    """
    for child in path_dir.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            folder_clean(child)
            child.rmdir()

if __name__ == "__main__":
    sysReset()
    


        