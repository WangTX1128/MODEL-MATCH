# -*- coding: utf-8 -*-
"""
Spyder Editor

Load data from excel 
"""
import pandas as pd
import glob
import numpy as np


class Load_Data(object):
    
    
    def __init__(self,path,data_type,sheet_name):
        self.sheet_name = sheet_name
        self.path = path
        self.data_type = data_type
    
    def Normalization(self, isp):
        spectrum_data = isp
        max = spectrum_data.max()
        min = spectrum_data.min()
        spectrum_norm = (spectrum_data - min)/(max - min)
        return spectrum_norm
    
    def from_excel(self):
        batch_data_path = glob.glob(self.path+'\*')
        object_data_path = [i for i in batch_data_path if self.data_type in i]
        if self.data_type == 'train':
            train_label = [i for i in object_data_path if 'event' in i][0]
            label_data = pd.read_excel(train_label, sheet_name = self.sheet_name, index_col = 1, header=None).values
            signal_data = pd.read_excel(train_label, sheet_name = self.sheet_name, index_col = 0, header=None).values
            ini_label = label_data[0]
            ini_signal = signal_data[0]
            round_label = np.zeros([13,1])
            round_signal = np.zeros([13,1])
            train_data_path = [i for i in object_data_path if 'data' in i][0]
            train_data = pd.read_excel(train_data_path, sheet_name = self.sheet_name, header=None).values
            round_train_data = np.zeros([13,20])
            for i in range(len(label_data)):
                round_label[i % 13] = label_data[i]
                round_signal[i % 13] = signal_data[i]
                round_train_data[i % 13] = train_data[signal_data[i]-1]
                if i % 13 == 0 and i != 0:
                    round_label[0]= label_data[0]
                    round_signal[0] = signal_data[0]
                    round_train_data[0] = train_data[signal_data[0]-1]
                    yield round_label[1:],round_signal[1:],round_train_data[1:]
        else:
            train_label = [i for i in object_data_path if 'event' in i][0]
            label_data = pd.read_excel(train_label, sheet_name = self.sheet_name, index_col = 1, header=None).values
            signal_data = pd.read_excel(train_label, sheet_name = self.sheet_name, index_col = 0, header=None).values
            ini_label = label_data[0]
            ini_signal = signal_data[0]
            round_label = np.zeros([13,1])
            round_signal = np.zeros([13,1])
            train_data_path = [i for i in object_data_path if 'data' in i][0]
            train_data = pd.read_excel(train_data_path, sheet_name = self.sheet_name, header=None).values
            round_train_data = np.zeros([13,20])
            for i in range(len(label_data)):
                round_label[i % 13] = label_data[i]
                round_signal[i % 13] = signal_data[i]
                round_train_data[i % 13] = train_data[signal_data[i]-1]
                if i % 13 == 0 and i != 0:
                    round_label[0]= label_data[0]
                    round_signal[0] = signal_data[0]
                    round_train_data[0] = train_data[signal_data[0]-1]
                    yield round_label[1:],round_signal[1:],self.Normalization(round_train_data[1:])
                

if __name__ == '__main__':
    path = r'D:\PycharmProjects\脑电信号分析\data\S1'
    data_type = 'test'
    sheet_name = 0
    preprocess = Load_Data(path,data_type,sheet_name)
    for i in preprocess.from_excel():
        print(i[2].max(),i[2].min())
        
