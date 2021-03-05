import sys
sys.path.append('F:\code\python\data_mining\possc-pytorch')

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from models.bp_network import BpNet
from constants.parameters import Parameters
from utils.steelmakingData import SteelmakingData, SMLoaderTrain


class BpModelTrain():
    def __init__(self, steelType: str, stoveNum: int, input_factors: list, output_factors: list):
        self.steelType = steelType
        self.stoveNum = stoveNum
        self.input_factors = input_factors
        self.output_factors = output_factors

        self.pm = Parameters()
        self.smlt = SMLoaderTrain(steelType, stoveNum, input_factors, output_factors)

        if torch.cuda.is_available():
            print("GPU available.")
        else:
            print("GPU not available. running with CPU.")
        print(Path.cwd())
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.bpNet = BpNet(len(self.input_factors), len(self.output_factors)).to(self.device)
        self.model_optimizer = optim.Adam(self.bpNet.parameters(), lr=self.pm.lr, betas=(0.9, 0.99))
        self.loss = nn.MSELoss()
        self.code_16, self.train_loader, self.val_loader = self.smlt.getTrainLoader()
        self.modelfolder_path = Path.cwd() / 'models' / 'models_trained' / self.steelType
        self.modelfolder_path.mkdir(parents=True, exist_ok=True)
        self.model_path = self.modelfolder_path / Path('stove' + str(self.stoveNum) + '#' + self.code_16 + '.pth')
        self.loss_his_train = []
        self.loss_his_val = []

    def set_train(self):
        """Convert models to training mode
        """
        self.bpNet.train()

    def set_eval(self):
        """Convert models to testing/evaluation mode
        """
        self.bpNet.eval()

    def train(self):
        self.epoch = 0
        for self.epoch in range(self.pm.epochs):
            loss_trained, loss_valed = self.run_epoch()
            self.loss_his_train.append(loss_trained)
            self.loss_his_val.append(loss_valed)
            if self.epoch % 10 == 0:
                print("Epoch:{}".format(self.epoch))
                print("train loss: {}".format(loss_trained))
                print("val loss: {}".format(loss_valed))
        print('训练完成，模型名为:%s' %(self.code_16))
        torch.save(self.bpNet.state_dict(), self.model_path)

    def run_epoch(self):
        loss_train = torch.tensor(.0).to(self.device)
        loss_val = torch.tensor(.0).to(self.device)

        self.set_train()
        for step, (batch_x, batch_y) in enumerate(self.train_loader):
            batch_x = torch.as_tensor(batch_x, dtype=torch.float32)
            batch_y = torch.as_tensor(batch_y, dtype=torch.float32)
            b_x = batch_x.to(self.device)
            outputs = self.bpNet(b_x)
            loss_step = torch.tensor(.0).to(self.device)
            for i in range(len(self.output_factors)):           
                loss_everyfactor = self.loss(outputs[:, i], batch_y[:, i].to(self.device))
                loss_step += loss_everyfactor
            self.model_optimizer.zero_grad()
            loss_step.backward()
            self.model_optimizer.step()
            loss_train += loss_step
        self.set_eval()
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(self.val_loader):
                loss_all = None
                batch_x = torch.as_tensor(batch_x, dtype=torch.float32)
                batch_y = torch.as_tensor(batch_y, dtype=torch.float32)
                b_x = batch_x.to(self.device)
                outputs = self.bpNet(b_x)
                loss_all = torch.tensor(.0).to(self.device)
                for i in range(len(self.output_factors)):
                    loss = self.loss(outputs[:, i], batch_y[:, i].to(self.device))
                    loss_all += loss
                loss_val += loss_all
        
        loss_trained = loss_train.cpu().detach().numpy()
        loss_valed = loss_val.cpu().detach().numpy()
        return loss_trained, loss_valed

    def plot_hisloss(self):
        plt.title('Loss')
        plt.plot(self.loss_his_train, label='train_loss', color='deepskyblue')
        plt.plot(self.loss_his_val, label='val_loss', color='pink')
        plt.legend(['train_loss', 'val_loss'])
        plt.show()


if __name__ == '__main__':
    pm = Parameters()
    bptrain = BpModelTrain('Q235B-Z', 1, pm.input_factorsTest, pm.output_factorsTest)
    bptrain.train()
    bptrain.plot_hisloss()