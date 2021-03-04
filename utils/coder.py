import os
import sys
import json
import sys
sys.path.append('F:\code\python\data_mining\possc-pytorch')
import numpy as np
from pathlib import Path
from constants.parameters import Parameters


class Coder():
    def __init__(self):
        self.pm = Parameters()
        self.input_factorsAll = self.pm.input_factorsAll
        self.output_factorsAll= self.pm.output_factorsAll
        self.maxlen_x = 40
        self.maxlen_y = 12
        self.dict_coder = {}

    def encoder_16(self, input_factors: list, output_factors: list) -> list:
        """
        将输入输出参数编码为16进制，编码方式为改进的onehot编码，先用若干位的二进制标记是否
        含有该参数，"1"表示有，随后通过每四位二进制合并为一个16进制的方式获得16进制编码
        """
        code_16 = []
        indexs_binX = np.zeros(self.maxlen_x).astype(int)
        indexs_binY = np.zeros(self.maxlen_y).astype(int)
        for i, val in enumerate(self.input_factorsAll):
            if val in input_factors:
                indexs_binX[i] = 1
        for i, val in enumerate(self.output_factorsAll):
            if val in output_factors:
                indexs_binY[i] = 1
        for i in range(int(self.maxlen_x / 4)):
            temp_10 = 8 * indexs_binX[4 * i] + 4 * indexs_binX[4 * i + 1] + 2 * indexs_binX[4 * i + 2] + indexs_binX[4 * i + 3]
            temp_16 = hex(temp_10)
            code_16.append(temp_16)
        for i in range(int(self.maxlen_y / 4)):
            temp_10 = 8 * indexs_binY[4 * i] + 4 * indexs_binY[4 * i + 1] + 2 * indexs_binY[4 * i + 2] + indexs_binY[4 * i + 3]
            temp_16 = hex(temp_10)
            code_16.append(temp_16)
        return code_16

    def encoder(self, path_jsonfile: Path, input_factors: list, output_factors: list) -> str:
        """
        编码函数，使用一个本地json文件保存编码字典，其key为两位16进制码，value为若干位16进制码，
        该value为encoder16函数返回的16进制编码，该key为最后模型名所跟的两位16进制码
        """
        if not path_jsonfile.is_file():
            json_newfile = open(path_jsonfile, 'w')
            json_newfile.close()
        else:
            with open(path_jsonfile, 'r+', encoding='gbk') as json_file:
                self.dict_coder = json.load(json_file)
                json_file.close()
        code_key = hex(len(self.dict_coder))
        code_value = self.encoder_16(input_factors, output_factors)
        if code_value not in self.dict_coder.values():
            self.dict_coder[code_key] = code_value
            json_str = json.dumps(self.dict_coder)
            with open(path_jsonfile, 'r+', encoding='gbk') as json_file:
                json_file.write(json_str)
                json_file.close()
        else:
            keys = filter(lambda x:code_value == x[1], self.dict_coder.items())
            for (key,value) in keys:
                code_key = key
        return code_key

    def decode_16(self, num16_list: list):
        """
        16进制解码函数，将一个若干16进制编码解码为输入和输出参数
        """
        # codeList列表用于存放解码后的二进制码
        codeList_input = []
        codeList_output = []
        # index列表用于存放与codeList对应的属性信息
        index_input = []
        index_output = []
        # outNum: 输出属性所占的16进制编码位数
        outNum = int(self.maxlen_y / (-4))
        input_list = num16_list[:outNum]
        output_list = num16_list[outNum:]
        for iter_ in input_list:
            code_2 = bin(int(iter_, 16))[2:]
            code_str = str(code_2)           
            for c in code_str:
                codeList_input.append(c)
        for i, val in enumerate(codeList_input):
            if val == '1':
                index_input.append(self.input_factorsAll[i])
        for iter_ in output_list:
            code_2 = bin(int(iter_, 16))[2:]
            code_str = str(code_2)           
            for c in code_str:
                codeList_output.append(c)
        for i, val in enumerate(codeList_output):
            if val == '1':
                index_output.append(self.output_factorsAll[i])
        return index_input,index_output

    def decoder(self, path_jsonfile: Path, model_key: str):
        """
        解码函数，用于将模型后所带的16进制码解码为输入输出参数
        """
        with open(path_jsonfile, 'r', encoding='gbk') as json_file:
            self.dict_coder = json.load(json_file)
            json_file.close()
        values = self.dict_coder[model_key]
        return self.decode_16(values)

if __name__ == '__main__':
    json_path = Path.cwd() / 'test.json'
    co = Coding()
    code_16 = co.encoder(json_path, co.pm.input_factorsTest, co.pm.output_factorsTest)
    input_factors, output_factors = co.decoder(json_path, code_16)
    print(code_16)
    print(input_factors)
    print(output_factors)
