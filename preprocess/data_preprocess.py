import sys
sys.path.append('F:\code\python\data_mining\possc-pytorch')

from constants.parameters import Parameters
from preprocess.s3_split import Split

class DataPreprocess(Split):
    def __init__(self, steelType: str, stoveNum: int, input_factors: list, output_factors: list):
        super().__init__(steelType, stoveNum, input_factors, output_factors)

    def dataPreprocess(self):
        code_16 = self.split()
        return code_16


if __name__ == '__main__':
    pm = Parameters()
    dp = DataPreprocess('Q235B-Z', 1, pm.input_factorsTest, pm.output_factorsTest)
    print(dp.dataPreprocess())
