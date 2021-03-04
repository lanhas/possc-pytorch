from pathlib import Path
from generation.s2_cleanout import Cleanout


class Classify(Cleanout):
    def __init__(self, folder_path: Path):
        super().__init__(folder_path)
    
    def classify(self):
        df_data = self.cleanout()
        df_data = df_data.set_index(['steel_type'], drop=False)
        df_data = df_data.sort_index()
        return df_data
