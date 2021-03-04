import sys
sys.path.append('F:\code\python\data_mining\possc-pytorch')

import torch
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
from models.bp_network import BpNet
from constants.parameters import Parameters
from utils.coder import Coder
from utils.steelmakingData import SMLoaderTest
from sklearn.preprocessing import MinMaxScaler


class BpModelTest():
    def __init__(self, steelType: str, stoveNum: int, code_16):
        self.steelType = steelType
        self.stoveNum = stoveNum
        self.code_16 = code_16
        self.coder = Coder()
        self.testDataPath = Path.cwd() / 'dataset' / 'test' / self.steelType / Path('stove' + str(self.stoveNum) + '#' + str(self.code_16) + '.csv')
        self.path_jsonfile = Path.cwd() / 'models' / 'models_coder' / self.steelType / Path('stove' + str(self.stoveNum) + '.json')
        self.path_scaler = Path.cwd() / 'models' / 'models_scaler' / self.steelType / Path('stove' + str(self.stoveNum) + '#' + str(self.code_16) + '.pkl')
        self.input_factors, self.output_factors = self.coder.decoder(self.path_jsonfile, self.code_16)

        self.pm = Parameters()
        self.scaler = joblib.load(self.path_scaler)
        self.smlt = SMLoaderTest(self.steelType, self.stoveNum, self.input_factors, self.output_factors)
        self.test_loader = self.smlt.getTestLoader(self.code_16)

        if torch.cuda.is_available():
            print("GPU available.")
        else:
            print("GPU not available. running with CPU.")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bpNet = BpNet(len(self.input_factors), len(self.output_factors)).to(self.device)
        self.modelfolder_path = Path.cwd() /'models' / 'models_trained' / self.steelType
        self.model_path = self.modelfolder_path / Path('stove' + str(self.stoveNum) + '#' + self.code_16 + '.pth')
        self.data_pred = []

    def set_eval(self):
        """Convert models to testing/evaluation mode
        """
        self.bpNet.eval()

    def test(self):
        if not Path.is_file(self.model_path):
            print('无该模型，请先进行训练！')
        else:
            self.bpNet.load_state_dict(torch.load(self.model_path))
            self.set_eval()
            for step,(batch_x, batch_y) in enumerate(self.test_loader):
                batch_x = torch.as_tensor(batch_x, dtype=torch.float32).to(self.device)
                batch_y = torch.as_tensor(batch_y, dtype=torch.float32)
                output_tensor = self.bpNet(batch_x)
                output = output_tensor.cpu().detach().numpy()
                output = self.inverse_transform(output)
                output = np.squeeze(output, 0)
                self.data_pred.append(output)
        self.test_print(self.data_pred)
        self.plot_result(self.data_pred)

    def inverse_transform(self, output_list):
        output_list = self.scaler.inverse_transform(output_list)
        return output_list

    def test_print(self, output_list):
        print("模型%s预测结果:" %(self.code_16))
        for i, v in enumerate(output_list):
            print("第%d个样本预测结果：" %(i))
            for j, val in enumerate(self.output_factors):
                print(val, ':', output_list[i][j])

    def plot_result(self, output_list):
        plot_pos = [221, 222, 223, 224]
        data_predicted = pd.DataFrame(output_list)
        data_original = pd.read_csv(self.testDataPath, encoding='gbk')
        data_original = data_original.iloc[:, len(self.input_factors):]
        x = range(len(output_list))
        fig = plt.figure(1)
        for i, val in enumerate(self.output_factors):
            ax = fig.add_subplot(plot_pos[i])
            plt.title('%s成分预测' %(val))
            plt.plot(x, data_predicted.iloc[:, i], label='predict', color='deepskyblue')
            plt.plot(x, data_original.iloc[:, i],  label='original', color='pink')
            plt.legend(['predict', 'original'])
        plt.show()
        

if __name__ == '__main__':
    pm = Parameters()
    bptest = BpModelTest('Q235B-Z', 1, '0x0')
    bptest.test()
