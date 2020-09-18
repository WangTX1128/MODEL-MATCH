# -*- coding: utf-8 -*-
"""
Spyder Editor

Load data from excel 
%2020/9/18 13:23 添加数据排序，标签01编码功能
%使用方法 is_training = 1 ,data_type = train返回的第一个值是编码后的标签
          is_training = 0,data_type = test 返回无标签数据
"""
import pandas as pd
import glob
import numpy as np


class Load_Data(object):
    
    
    def __init__(self,path,data_type,sheet_name,is_training):
        self.sheet_name = sheet_name
        self.path = path
        self.data_type = data_type
        self.is_training = is_training
        
    def Normalization(self, isp):
        spectrum_data = isp
        max = spectrum_data.max()
        min = spectrum_data.min()
        spectrum_norm = (spectrum_data - min)/(max - min)
        return spectrum_norm
    
    def from_excel(self):
        batch_data_path = glob.glob(self.path+'\*')
        object_data_path = [i for i in batch_data_path if self.data_type in i]
        sheet_list = {}
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
                    sorted_label = np.zeros([12,1])
                    sorted_signal =np.zeros([12,1])
                    sorted_train_data =np.zeros([12,20])
                    count = 0
                    for i in round_label[1:]:
                        index = np.where(round_label[1:]==count+1)
                        sorted_label[count] = round_label[1:][index]
                        sorted_signal[count] = round_signal[1:][index]
                        sorted_train_data[count] = round_train_data[1:][index[0]]
                        count += 1
                    if self.is_training == 1:
                        sheet_list = {'char01(B)':(1,8),'char02(D)':(1,10),'char03(G)':(2,7),'char04(L)':(2,12),
                                      'char05(O)':(3,9),'char06(Q)':(3,11),'char07(S)':(4,7),'char08(V)':(4,10),
                                      'char09(Z)':(5,8),'char10(4)':(5,12),'char11(7)':(6,9),'char12(9)':(6,11)}
                        x = sheet_list[self.sheet_name][0]
                        y = sheet_list[self.sheet_name][1]
                        codeing_label = np.zeros([12,1])
                        codeing_label[x-1] = 1
                        codeing_label[y-1] = 1
                        yield codeing_label,sorted_label,sorted_signal,self.Normalization(sorted_train_data)
                    else:
                        yield sorted_label,sorted_signal,self.Normalization(sorted_train_data)
        else:
            train_label = [i for i in object_data_path if 'event' in i][0]
            try:
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
                        sorted_label = np.zeros([12,1])
                        sorted_signal =np.zeros([12,1])
                        sorted_train_data =np.zeros([12,20])
                        count = 0
                        for i in round_label[1:]:
                            index = np.where(round_label[1:]==count+1)
                            sorted_label[count] = round_label[1:][index]
                            sorted_signal[count] = round_signal[1:][index]
                            sorted_train_data[count] = round_train_data[1:][index[0]]
                            count += 1
                        yield sorted_label,sorted_signal,self.Normalization(sorted_train_data)
            except Exception as e:
                print('warning:测试集无标签，如想读取数据，请将sheet_name改为0,1,2等数字来索引')

if __name__ == '__main__':
    #path = r'D:\PycharmProjects\脑电信号分析\data\S1'
    path = r'D:\PycharmProjects\脑电信号分析\data\S1'
    data_type = 'train'
    sheet_name = 'char01(B)'
    #sheet_name = 0
    preprocess = Load_Data(path,data_type,sheet_name,is_training = 1)
    for i in preprocess.from_excel():
        #print(i[2].max(),i[2].min())
        print(i[0])
        break
        
