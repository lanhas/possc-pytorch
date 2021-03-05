import sys
sys.path.append('F:\code\python\data_mining\possc-pytorch')

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from constants.parameters import Parameters
from utils.coder import Coder
from utils.steelmakingData import SMLoaderRegressionm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


class LRegressionPred():
    def __init__(self, steelType: str, stoveNum: int, input_factors: list, output_factors: list, input_factorsNum):
        self.steelType = steelType
        self.stoveNum = stoveNum
        self.input_factors = input_factors
        self.output_factors = output_factors
        self.input_factorsNum = input_factorsNum

        self.pm = Parameters()
        self.smlt = SMLoaderRegressionm(self.steelType, self.stoveNum, self.input_factors, self.output_factors)
        self.input_regression, self.output_regression = self.smlt.getRegressionLoader()
        self.model = LinearRegression()

    def linearRegression(self):
        input_values = []
        for i, v in enumerate(self.input_factorsNum):
            if v != '':
                input_values.append(float(v))
        input_values = np.array(input_values)
        input_values = input_values[:, np.newaxis]
        self.model.fit(self.input_regression, self.output_regression)
        intercept = self.model.intercept_
        coef = self.model.coef_
        output_predict = self.model.predict(input_values.T)
        self.lr_print(intercept, coef)
        print('预测结果：', output_predict)

    def lr_print(self, intercept, coef):
        res = self.output_factors[0] + '='
        for i, v in enumerate(self.input_factors):
            res = res + str(coef[0][i]) + '*' + str(self.input_factors[i]) + '+'
        res = res + str(intercept[0])
        print("最佳拟合线:截距",intercept[0],",回归系数：",coef[0])
        print('-----------------------------------------------------------------------')
        print(res,'\n')
        
if __name__ == '__main__':
    pm = Parameters()
    input_factorNum = [0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1, 0.1]
    lrp = LRegressionPred('Q235B-Z', 1, pm.input_factorsTest, pm.output_factorsRegression, input_factorNum)
    lrp.linearRegression()







