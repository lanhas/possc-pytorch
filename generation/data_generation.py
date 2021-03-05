import sys
sys.path.append('F:\code\python\data_mining\possc-pytorch')

from pathlib import Path
from generation.s3_classify import Classify


class DataGeneration():
    def __init__(self):
        self.source_path = Path.home() / 'SteelmakingData'
        self.target_path = Path.cwd() / 'dataset' / 'native'
    
    def dataGeneration(self):
        source_subpath = self.source_path.iterdir()
        for _iter in source_subpath:
            subfolder_name = _iter.name
            native_ = Classify(_iter)
            native_dfdata = native_.classify()
            native_dfdata.to_csv(Path(self.target_path, subfolder_name + '.csv'), encoding='gbk', index=0)
        
if __name__ == '__main__':
    da = DataGeneration()
    da.dataGeneration()
