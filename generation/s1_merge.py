import pandas as pd
from pathlib import Path
from generation.s0_trans import Trans


class Merge(Trans):
    def __init__(self, folder_path: Path):
        super().__init__(folder_path)

    def merge_rows(self) -> pd.DataFrame:
        df_list = self.trans_columns()
        res_data = pd.concat(df_list, axis=0)
        return res_data



