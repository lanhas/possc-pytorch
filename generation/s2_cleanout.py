import re
import time
import pandas as pd
from pathlib import Path
from generation.s1_merge import Merge


class Cleanout(Merge):
    def __init__(self, folder_path: Path):
        super().__init__(folder_path)

    def transforTime(self, time):
        if re.search('\d[:]\d', str(time)):
            lists = str(time).split(':',1)
            res = float(lists[0]) + float(lists[-1])/60
            return round(res, 2)
        else:
            return None
    
    def cleanout_time(self, df_data) -> pd.DataFrame:
        res_data = df_data
        indexs_time = self.factors.factors_time
        for factor in indexs_time:
            res_data[factor] = res_data[factor].map(lambda x: self.transforTime(x))
        return res_data

    def transforPercent(self, obj):
        if re.search('\d[%]', str(obj)):
            lists = str(obj).split('%', 1)
            res = float(lists[0])
            return round(res, 2)
        else:
            return None

    def cleanout_percent(self, df_data) -> pd.DataFrame:
        res_data = df_data
        indexs_percent = self.factors.factors_percent
        for factor in indexs_percent:
            res_data[factor] = res_data[factor].map(lambda x: self.transforPercent(x))
        return res_data 

    def cleanout_symbol(self, df_data) -> pd.DataFrame:
        res_data = df_data
        indexs_symbol = self.factors.factors_symbol
        for factor in indexs_symbol:
            res_data[factor] = res_data[factor].apply(pd.to_numeric, errors='coerce')
        return res_data
        
    def cleanout(self) -> pd.DataFrame:
        df_data = self.merge_rows()
        df_data = self.cleanout_time(df_data)
        df_data = self.cleanout_percent(df_data)
        res_data = self.cleanout_symbol(df_data)
        return res_data
