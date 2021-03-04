import pandas as pd
from pathlib import Path
from constants.parameters import SmeltFactors

class Trans():
    def __init__(self, folder_path: Path):
        self.folder_path = folder_path
        self.factors = SmeltFactors()

    def extract_columns(self, df_data, columns) -> pd.DataFrame:
        df_list = []
        for i in columns:
            df_list.append(df_data[i])
        res_data = pd.concat(df_list, axis=1)
        return res_data

    def trans_columns(self) -> list:
        df_list = []
        for path_smeltData in self.folder_path.iterdir():
            df_data = pd.read_csv(path_smeltData, encoding='gbk')
            df_data = self.extract_columns(df_data, self.factors.factors_zh)
            df_data.columns = self.factors.factors_en
            df_list.append(df_data)
        return df_list



