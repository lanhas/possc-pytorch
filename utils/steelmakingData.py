import sys
sys.path.append('F:\code\python\data_mining\possc-pytorch')
from constants.parameters import Parameters

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from preprocess.s1_cleanout import Cleanout
from sklearn.model_selection import train_test_split
from preprocess.data_preprocess import DataPreprocess

class SteelmakingData(Dataset):
    def __init__(self, df_data: pd.DataFrame, len_inputFactors: int, transform=None):
        self.len_inputFactors = len_inputFactors
        self.transform = transform
        self.df_data = df_data

    def __len__(self):
        return self.df_data.shape[0]

    def __getitem__(self, index):
        sample = (np.array(self.df_data.iloc[index, :self.len_inputFactors]),
                 np.array(self.df_data.iloc[index, self.len_inputFactors:]))
        return sample


class SMLoaderTrain(DataPreprocess):
    def __init__(self, steelType: str, stoveNum: int, input_factors: list, output_factors: list):
        super().__init__(steelType, stoveNum, input_factors, output_factors)

    def getTrainLoader(self):
        code_16 = self.get_code16()
        trainDataPath = Path.cwd() / 'dataset' / 'train' / self.steelType / Path('stove' + str(self.stoveNum) + '#' + str(code_16) + '.csv')
        # if not Path.is_file(trainDataPath):
        self.dataPreprocess()
        df_data = pd.read_csv(trainDataPath, encoding='gbk')
        df_train, df_val = train_test_split(df_data, test_size=self.pm.val_size, random_state=42)
        data_train = SteelmakingData(df_train, len(self.input_factors))
        data_val = SteelmakingData(df_val, len(self.input_factors))
        dataloader_train = DataLoader(data_train, batch_size=self.pm.batch_size, shuffle=True)
        dataloader_val = DataLoader(data_val, batch_size=self.pm.batch_size, shuffle=True)
        return code_16, dataloader_train, dataloader_val

class SMLoaderTest(DataPreprocess):
    def __init__(self, steelType: str, stoveNum: int, input_factors: list, output_factors: list):
        super().__init__(steelType, stoveNum, input_factors, output_factors)
    
    def getTestLoader(self, code_16: str):
        testDataPath = Path.cwd() / 'dataset' / 'test' / self.steelType / Path('stove' + str(self.stoveNum) + '#' + str(code_16) + '.csv')
        if not Path.is_file(testDataPath):
            self.dataPreprocess()
        df_data = pd.read_csv(testDataPath, encoding='gbk')
        data_test = SteelmakingData(df_data, len(self.input_factors))
        dataloader_test = DataLoader(data_test, batch_size=1)
        return dataloader_test

class SMLoaderPred():
    def __init__(self, input_factors: list, input_factorsNum: list):
        self.input_factors = input_factors
        self.input_factorsNum = input_factorsNum
        self.input_values = []

    def getPredLoader(self):
        for i, v in enumerate(self.input_factorsNum):
            if v != '':
                self.input_values.append(float(v))
        input_values = np.array(self.input_values)
        input_values = input_values[:, np.newaxis]
        values_torch = torch.as_tensor(torch.from_numpy(input_values),dtype=torch.float32).T
        # values_torch = torch.as_tensor(values_torch, dtype=torch.float32)
        return values_torch

class SMLoaderRegressionm(Cleanout):
    def __init__(self, steelType: str, stoveNum: int, input_factors: list, output_factors: list):
        super().__init__(steelType, stoveNum, input_factors, output_factors)

    def getRegressionLoader(self):
        df_data = self.cleanout()
        df_inputfactors = df_data.iloc[:, : len(self.input_factors)]
        df_outputfactors = df_data.iloc[:, len(self.input_factors):]
        return df_inputfactors, df_outputfactors

if __name__ == '__main__':
    pm = Parameters()
    smlt = SMLoaderTrain('Q235B-Z', 1, pm.input_factorsTest, pm.output_factorsTest)
    code_16, dataloader_train, dataloader_val = smlt.getTrainLoader()
    for i_batch, (batch_x, batch_y) in enumerate(dataloader_val):
        print(i_batch)
        print(batch_x)