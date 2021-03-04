from pathlib import Path


class Clean():
    def __init__(self, steelType, stoveNum, code_16 = None):
        self.steelType = steelType
        self.stoveNum = stoveNum
        self.code_16 = code_16
        self.dir_trainFile = Path.cwd() / 'dataset' / 'train' / self.steelType
        self.dir_testFile = Path.cwd() / 'dataset' / 'test' / self.steelType
        self.dir_modelsScaler = Path.cwd() / 'models' / 'models_scaler' / self.steelType
        self.dir_modelsCoder = Path.cwd() / 'models' / 'models_coder' / self.steelType
        self.dir_modelsTrained = Path.cwd() / 'models' / 'models_trained' / self.steelType
        
    def rm_coderModel(self):
        fileName =  Path('stove' + str(self.stoveNum) +'.json')
        path_modelsCoder = self.dir_modelsCoder / fileName
        try:
            path_modelsCoder.unlink()
            print('编码器文件%s删除成功！' %(str(fileName)))
        except Exception  as e:
            print('编码器文件%s删除失败，请排查！' %(str(fileName)))
            print(type(e),e)
    
    def rm_trainedModel(self):
        fileName = Path('stove' + str(self.stoveNum) + '#' + self.code_16 + '.pth')
        path_modelTrained = self.dir_modelsTrained / fileName
        try:
            path_modelTrained.unlink()
            print('模型文件%s删除成功！' %(str(fileName)))
        except Exception  as e:
            print('模型文件%s删除失败，请排查！' %(str(fileName)))
            print(type(e),e)

    def rm_scalerModel(self):
        fileName = Path('stove' + str(self.stoveNum) + '#' + self.code_16 + '.pkl')
        path_modelScaler= self.dir_modelsScaler / fileName
        try:
            path_modelScaler.unlink()
            print('Scaler文件%s删除成功！' %(str(fileName)))
        except Exception  as e:
            print('Scaler文件%s删除失败，请排查！' %(str(fileName)))
            print(type(e),e)

    def rm_dataFile(self):
        fileName = Path('stove' + str(self.stoveNum) + '#' + self.code_16 + '.csv')
        path_trainFile= self.dir_trainFile / fileName
        path_testFile= self.dir_testFile / fileName
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

    def rm_steelTypeFile(self, code_16):
        self.code_16 = code_16
        self.rm_dataFile()
        self.rm_scalerModel()
        self.rm_trainedModel()

    def folder_clean(self, folder_path):
        for child in folder_path.glob('*'):
            if child.is_file():
                child.unlink()
            else:
                self.folder_clean(child)

    def rm_steelTypeAllFile(self):
        try:
            self.folder_clean(self.dir_trainFile)
            self.folder_clean(self.dir_testFile)
            self.folder_clean(self.dir_modelsScaler)
            self.folder_clean(self.dir_modelsCoder)
            self.folder_clean(self.dir_modelsTrained)
            print('钢种类型%s下所有数据删除成功！' %(self.steelType))
        except Exception  as e:
            print('钢种类型%s下所有数据删除失败！' %(self.steelType))
            print(type(e), e)


class SystemReset():
    def __init__(self):
        self.dir_trainFile = Path.cwd() / 'dataset' / 'train'
        self.dir_testFile = Path.cwd() / 'dataset' / 'test'
        self.dir_modelsScaler = Path.cwd() / 'models' / 'models_scaler'
        self.dir_modelsCoder = Path.cwd() / 'models' / 'models_coder'
        self.dir_modelsTrained = Path.cwd() / 'models' / 'models_trained'

    def reset(self):
        try:
            self.folder_clean(self.dir_trainFile)
            self.folder_clean(self.dir_testFile)
            self.folder_clean(self.dir_modelsScaler)
            self.folder_clean(self.dir_modelsCoder)
            self.folder_clean(self.dir_modelsTrained)
            print('系统重置成功')
        except Exception as e:
            print('系统重置失败！请排查！')
            print(type(e), e)

    def folder_clean(self, folder_path):
        for child in folder_path.glob('*'):
            if child.is_file():
                child.unlink()
            else:
                self.folder_clean(child)
                child.rmdir()

if __name__ == "__main__":
    reset = SystemReset()
    reset.reset()
    


        