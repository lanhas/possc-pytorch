import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from preprocess.s1_cleanout import Cleanout


class Normalization(Cleanout):
    def __init__(self, steelType: str, stoveNum: int, input_factors: list, output_factors: list):
        super().__init__(steelType, stoveNum, input_factors, output_factors)
    
    def normorlize(self, path_scalerfile):
        df_data = self.cleanout()
        
        df_inputfactors = df_data.iloc[:, :len(self.input_factors)]
        df_outputfactors = df_data.iloc[:, len(self.input_factors):]
        scaler_input = MinMaxScaler()
        scaler_output = MinMaxScaler()
        data_inputfactors = scaler_input.fit_transform(df_inputfactors)
        data_outputfactors = scaler_output.fit_transform(df_outputfactors)
        joblib.dump(scaler_output, path_scalerfile)
        df_inputfactors = pd.DataFrame(data_inputfactors)
        df_outputfactors = pd.DataFrame(data_outputfactors)
        df_data = pd.concat((df_inputfactors, df_outputfactors), axis=1)
        return df_data


    