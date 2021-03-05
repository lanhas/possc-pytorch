import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.coder import Coder
from preprocess.s2_normalization import Normalization


class Split(Normalization):
    def __init__(self, steelType: str, stoveNum: int, input_factors: list, output_factors: list):
        super().__init__(steelType, stoveNum, input_factors, output_factors)

    def split(self) -> str:
        datasetFolder_train = Path.cwd() / 'dataset' / 'train' / self.steelType 
        datasetFolder_test = Path.cwd() / 'dataset' / 'test' / self.steelType
        scalerFolder = Path.cwd() / 'models' / 'models_scaler' / self.steelType
        datasetFolder_train.mkdir(parents = True, exist_ok = True)
        datasetFolder_test.mkdir(parents = True, exist_ok = True)
        scalerFolder.mkdir(parents=True, exist_ok=True)

        code_16 = self.get_code16()
        datasetPath_train = datasetFolder_train / Path('stove' + str(self.stoveNum) + '#' + str(code_16) + '.csv')
        datasetPath_test = datasetFolder_test / Path('stove' + str(self.stoveNum) + '#' + str(code_16) + '.csv')
        scalerPath = scalerFolder / Path('stove' + str(self.stoveNum) + '#' + str(code_16) + '.pkl')

        df_data = self.normorlize(scalerPath)
        df_train, df_test = train_test_split(df_data, test_size=self.pm.test_size, random_state=42)
        df_train.to_csv(datasetPath_train, encoding='gbk', index=0)
        df_test.to_csv(datasetPath_test, encoding='gbk', index=0)

        return str(code_16)



