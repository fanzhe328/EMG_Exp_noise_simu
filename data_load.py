# *-* coding: UTF-8 *-*
import os, sys
import numpy as np


root_path = os.getcwd()


def load_feature_dataset(dir='train1_250_100', subject='subject_1', feature_type='TD4', action_num=11):
    ''' 读取样本数据和类别数据， dir表示实验组名称，subject表示受试者，subject_all表示所有受试者'''
    file_path = root_path + '/' + dir + '/' + subject
    trains = np.load(file_path + '_feat_'+ feature_type + '_trains_1-'+str(action_num)+'.npy')
    classes = np.load(file_path + '_feat_'+ feature_type + '_classes_1-'+str(action_num)+'.npy')
    return trains, classes



if __name__ == '__main__':
    load_feature_dataset()