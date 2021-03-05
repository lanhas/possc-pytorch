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
from utils.steelmakingData import SMLoaderPred
from sklearn.preprocessing import MinMaxScaler


class BpModelPred():
    def __init__(self, steelType: str, stoveNum: int, input_factorsNum: list, code_16):
        self.steelType = steelType
        self.stoveNum = stoveNum
        self.input_factorsNum = input_factorsNum
        self.code_16 = code_16
        self.coder = Coder()
        self.path_coder = Path.cwd() / 'models' / 'models_coder' / self.steelType / Path('stove' + str(self.stoveNum) + '.json')
        self.path_scaler = Path.cwd() / 'models' / 'models_scaler' / self.steelType / Path('stove' + str(self.stoveNum) + '#' + str(self.code_16) + '.pkl')
        self.input_factors, self.output_factors = self.coder.decoder(self.path_coder, self.code_16)

        self.pm = Parameters()
        self.scaler = joblib.load(self.path_scaler)
        if torch.cuda.is_available():
            print("GPU available.")
        else:
            print("GPU not available. running with CPU.")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.smlt = SMLoaderPred(self.input_factors, self.input_factorsNum)
        self.pred_data = self.smlt.getPredLoader().to(self.device)

        self.bpNet = BpNet(len(self.input_factors), len(self.output_factors)).to(self.device)
        self.modelfolder_path = Path.cwd() /'models' / 'models_trained' / self.steelType
        self.model_path = self.modelfolder_path / Path('stove' + str(self.stoveNum) + '#' + self.code_16 + '.pth')
        self.data_pred = []

    def set_eval(self):
        """Convert models to testing/evaluation mode
        """
        self.bpNet.eval()

    def predict(self):
        self.bpNet.load_state_dict(torch.load(self.model_path))
        self.set_eval()
        output_tensor = self.bpNet(self.pred_data)
        output = output_tensor.cpu().detach().numpy()
        output = self.inverse_transform(output)
        output = np.squeeze(output, 0)
        self.pred_print(output)

    def inverse_transform(self, input_list):
        output_list = self.scaler.inverse_transform(input_list)
        return output_list

    def pred_print(self, input_list):
        print("模型%s预测结果:" %(self.code_16))
        for i, v in enumerate(self.output_factors):
            print(v, ':', input_list[i])


if __name__ == '__main__':
    pm = Parameters()
    input_factorNum = [0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1,0.1, 0.1]
    bptest = BpModelPred('Q235B-Z', 1, input_factorNum, '0x0')
    bptest.predict()