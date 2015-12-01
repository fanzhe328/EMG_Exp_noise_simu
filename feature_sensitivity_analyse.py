# *-* coding: UTF-8 *-*
import os
import sys
import numpy as np
import data_load
import matplotlib.pyplot as plt

root_path = os.getcwd()

from preprocess import load_raw_dataset

def new_fold(log_fold):
    if os.path.isdir(log_fold) == False:
        try:
            os.makedirs(log_fold)
        except:
            print "Can not create log fold! "
            return False
    return True

def feature_action_sensitivity(featuret_type='TD4', action_num=7):
    ''' 对每个特征，分析其在不移位和移位情况下的差异性 '''
    results = []
    channel_pos_list = ['S0',                                             # 中心位置
                        'U1', 'U2', 'D1', 'D2', 'L1', 'L2', 'R1', 'R2']  # 上 下 左 右
    subjects = ['subject_' + str(i + 1) for i in range(5)]
    feature_list = ['MAV', 'ZC', 'SSC', 'WL']
    actions = [i+1 for i in range(action_num)]
    groups = [i+1 for i in range(4)]

    channel_num = 4
    feat_num = len(feature_list)
    channel_span = channel_num*feat_num
    # print channel_span
    # sys.exit(0)
    results.append(['subject', 'action', 'feature', 'group'] + channel_pos_list[1:])
    for subject in subjects:
        for action in actions:
            filename = subject+'_feat_'+featuret_type+'_action_'+str(action)
            data = np.load(root_path + '/train4_250_100/' + filename + '.npy')
            means = np.mean(data, axis=0)
            stds = np.std(data, axis=0)

            for group in groups:
                for feat_idx, feat_name in enumerate(feature_list):
                    idx = np.array(
                        [(i+1)*channel_span+feat_idx+feat_num*(group-1)
                            for i in range(len(channel_pos_list)-1)])
                    idx_o = feat_num*(group-1)+feat_idx
                    # print idx, idx_o
                    # variation = abs(means[idx] - means[idx_o])/means[idx_o] \
                    #             + abs(stds[idx]-stds[idx_o])/stds[idx_o]
                    variation = abs(means[idx] - means[idx_o])/means[idx_o]
                    results.append([subject, str(action), feat_name, str(group)] + map(str, variation))
                    # print results
                    # print subject, action, feat_name, group, variation[:]
    fold_path = root_path + '/result/sensitivity/'
    new_fold(fold_path) 
    log_result(results, fold_path+'feature_action_sensitivity', 2)
                # sensitives = np.ones( ( channel_span-1,))
                # print sensitives.shape
                # stds = np.ones( (channel_span-1,))

def log_result(results, log_file, flag):
    np.save(log_file + '.npy', results)
    if flag == 2:
        np.savetxt(log_file + '.csv', results, fmt="%s", delimiter=",")

if __name__ == '__main__':
    feature_action_sensitivity()
