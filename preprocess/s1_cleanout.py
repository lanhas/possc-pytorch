import numpy as np
import pandas as pd
from pathlib import Path
from preprocess.s0_native import DataNative


class Cleanout(DataNative):
    def __init__(self, steelType: str, stoveNum: int, input_factors: list, output_factors: list):
        super().__init__(steelType, stoveNum, input_factors, output_factors)

    def drop_nan(self, df_data, drop_threshold):
        data = df_data
        label_drop = []
        for i in range(data.shape[0]):
            if data.iloc[i].isnull().values.sum() > drop_threshold:
                label_drop.append(i)
        data.index = range(len(data))
        droped_data = data.drop(label_drop, axis=0)
        # res_data = droped_data.fillna(droped_data.mean())
        for i in range(droped_data.shape[1]):
            droped_data.iloc[:, i] = droped_data.iloc[:, i].fillna(droped_data.iloc[:, i].mean())
        return droped_data

    def drop_error(self, df_data, drop_coef):
        error_rows = []
        for j in range(df_data.shape[1]):
            Percentile = np.percentile(df_data.iloc[:, j],[0,25,50,75,100])
            IQR = Percentile[3] - Percentile[1]
            UpLimit = Percentile[3]+IQR * drop_coef
            DownLimit = Percentile[1]-IQR * drop_coef
            for i in range(df_data.iloc[:, j].shape[0]):
                if df_data.iloc[i, j] > UpLimit or df_data.iloc[i, j] < DownLimit:
                    error_rows.append(i)
        error_rows = np.unique(error_rows)
        df_data.index = range(len(df_data))
        res_data = df_data.drop(error_rows, axis=0)
        return res_data

    def cleanout(self):
        df_data = self.extrateData()
        df_data = self.drop_nan(df_data, self.pm.drop_nan_coef)
        res_data = self.drop_error(df_data, self.pm.drop_err_coef)
        return res_data 
