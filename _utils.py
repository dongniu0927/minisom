# -*- coding: utf-8 -*-
from enum import Enum
import json, math, os
import numpy.linalg as LA  # 计算范数
import numpy as np

class Utils:

    msg_type = Enum("msg_type", ("info", "warning", "fail"))

    def __init__(self):
        pass

    @staticmethod
    def show_msg(msg, type=msg_type.info):
        print("[", type.name, "]", msg)

    @staticmethod
    def dump_json(filename, data_source):
        try:
            with open(filename, "w") as f:
                json.dump(data_source, f, sort_keys=True, indent=4, separators=(',', ': '))
                return True
        except :
            Utils.show_msg("存储json出错...")
            return False

    @staticmethod
    def load_json(filename):
        try:
            with open(filename) as f:
                return json.load(f)
        except :
            return None

    @staticmethod
    def normalize(v):  # v为向量
        norm = LA.norm(v, 2)  # 计算2范数
        v_new = []
        for i in range(len(v)):
            v_new.append(round(v[i] / norm, 2))  # 保留2位小数
        return v_new

    @staticmethod
    # 计算x,y两个向量的欧式距离
    def calSimilarity(x, y):
        if len(x) != len(y):
            raise("维度不一致无法计算欧式距离!")
        c = 0
        for i in range(len(x)):
            c += pow((x[i] - y[i]), 2)
        return math.sqrt(c)

    @staticmethod
    def duplicated_varnames(df):
        """Return a dict of all variable names that
        are duplicated in a given dataframe."""
        repeat_dict = {}
        var_list = list(df)  # list of varnames as strings
        for varname in var_list:
            # make a list of all instances of that varname
            test_list = [v for v in var_list if v == varname]
            # if more than one instance, report duplications in repeat_dict
            if len(test_list) > 1:
                repeat_dict[varname] = len(test_list)
        return repeat_dict

    @staticmethod
    def reshape_1d_to_2d(data_list, row, col):
        res = []
        for i in range(row):
            unit = []
            for j in range(col):
                unit.append(data_list[i*col+j])
            res.append(unit)
        return res

    @staticmethod
    def get_filenames(filepath):
        return os.listdir(filepath)

    @staticmethod
    def gen_scatter_from_vector(v):  # 通过vector生成scatter
        v = np.array(v).flatten()
        c = 0
        for i in v:
            c += i
        c = c*1.0/len(v)
        return c