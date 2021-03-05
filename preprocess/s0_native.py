import pandas as pd
from pathlib import Path
from utils.coder import Coder
from constants.parameters import Parameters


class DataNative():
    def __init__(self, steelType: str, stoveNum: int, input_factors: list, output_factors: list):
        self.steelType = steelType
        self.stoveNum = stoveNum
        self.input_factors = input_factors
        self.output_factors = output_factors
        self.pm = Parameters()

    def extract_columns(self, df_data) -> pd.DataFrame:
        columns = self.input_factors + self.output_factors
        df_list = []
        for i in columns:
            df_list.append(df_data[i])
        res_data = pd.concat(df_list, axis=1)
        return res_data

    def extrateData(self):
        fileName = 'stove' + str(self.stoveNum) + '.csv'
        steelmakingDataPath = Path.cwd() / 'dataset' / 'native' / fileName
        df_data = pd.read_csv(steelmakingDataPath, encoding='gbk')
        df_data = df_data.set_index(['steel_type'], drop=False)
        df_data = df_data.loc[self.steelType]
        res_data = self.extract_columns(df_data)
        return res_data
    
    def get_code16(self):
        coder = Coder()
        coderFolder = Path.cwd() / 'models' / 'models_coder' / self.steelType
        coderFolder.mkdir(parents=True, exist_ok=True)
        coderPath = coderFolder / Path('stove' + str(self.stoveNum) + '.json')
        code_16 = coder.encoder(coderPath, self.input_factors, self.output_factors)
        return code_16


