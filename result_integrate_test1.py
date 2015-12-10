# *-* coding: UTF-8 *-*
import os, sys, re, time
import numpy as np
from file_script import log_result, new_fold

root_path = os.getcwd()
channel_pos_list = ['S0',                                             # 中心位置
                    'U1', 'U2', 'D1', 'D2', 'L1', 'L2', 'R1', 'R2']  # 上 下 左 右
proportional_list = ['1.0', '0.9', '0.8', '0.7', '0.6', '0.8-0.9', '0.7-0.9', '0.6-0.9']
action_lists = [7, 9, 11]
# action_lists = [7]

def result_load(dir='250_100', feature_type='TD4', subject='subject_1', norm='_norm', action='7', training_type='intra'):
    file_path = root_path + '/result_test1/' + dir + '/' +\
                feature_type+'_data4_'+ subject + norm + '/' +\
                'feat_'+feature_type+'_'+training_type+'_action_1-'+str(action)+'.npy'
    
    data = np.load(file_path)
    return data

def result_integrate_intra(time_now):
    training_type = 'intra'
    span = len(proportional_list)
    fold_path = root_path + '/result_test1/proportional_integrate'
    new_fold(fold_path) 
    feature_type = 'TD4'
    norm = '_norm'

    subject_list = ['subject_' + str(i) for i in range(1, 6)]

    res_all = []
    blank_line = ['' for i in range(len(channel_pos_list))]
    res_all.append(blank_line)

    for action in action_lists:
        for subject in subject_list:
            res = []
            index = 2
            
            res_ind = 1

            data = result_load('250_100',feature_type, subject, norm, action, training_type)
            title = feature_type+'_'+subject+'_action_1-'+str(action)
            res_head = [title]
            res_head.extend(proportional_list)
            res.append(res_head)
            
            for i in range(len(channel_pos_list)):
                res_intra = [channel_pos_list[i]]
                # print res_intra
                res_intra.extend(map(float,data[index:index+span,4][:]))
                index += span
                res.append(res_intra)

            res_np = np.array(res)
            res_aver = ['average']
            for i in range(len(proportional_list)):
                res_aver.append(np.mean(map(float,res_np[res_ind:,i+1])))
            res.append(res_aver)
            # file_path = fold_path + '/prop_'+training_type+'_'+title+'_'+str(time_now)
            # log_result(res, file_path, 2)

            res_all.extend(res)
            res_all.append(blank_line)
            res_all.append(blank_line)
            res_all.append(blank_line)
            res_all.append(blank_line)
            res_all.append(blank_line)

    file_path = fold_path + '/prop_'+training_type+'_all_'+str(time_now)
    log_result(res_all, file_path, 2)

def result_integrate_inter(time_now):
    training_type = 'inter'
    span = len(proportional_list)
    fold_path = root_path + '/result_test1/proportional_integrate'
    new_fold(fold_path) 
    feature_type = 'TD4'
    norm = '_norm'
    subject_list = ['subject_' + str(i) for i in range(1, 6)]

    res_all = []
    blank_line = ['' for i in range(len(channel_pos_list))]
    res_all.append(blank_line)

    for action in action_lists:
        for subject in subject_list:
            res = []
            index = 2
            
            res_ind = 1

            data = result_load('250_100',feature_type, subject, norm, action, training_type)
            title = feature_type+'_'+subject+'_action_1-'+str(action)
            res_head = [title]
            res_head.extend(proportional_list)
            res.append(res_head)
            
            for i in range(len(channel_pos_list)-1):
                res_intra = [channel_pos_list[i+1]]
                # print res_intra
                res_intra.extend(map(float,data[index:index+span,4][:]))
                index += span
                res.append(res_intra)

            res_np = np.array(res)
            res_aver = ['average']
            for i in range(len(proportional_list)):
                res_aver.append(np.mean(map(float,res_np[res_ind:res_ind+8,i+1])))
            res.append(res_aver)
            # file_path = fold_path + '/prop_'+training_type+'_'+title+'_'+str(time_now)
            # log_result(res, file_path, 2)

            res_all.extend(res)
            res_all.append(blank_line)
            res_all.append(blank_line)
            res_all.append(blank_line)
            res_all.append(blank_line)
            res_all.append(blank_line)

    file_path = fold_path + '/prop_'+training_type+'_all_'+str(time_now)
    log_result(res_all, file_path, 2)

if __name__ == '__main__':
    time_now = time.strftime('%Y-%m-%d_%H-%M',time.localtime(time.time()))
    # print time_now
    result_integrate_intra(time_now)
    result_integrate_inter(time_now)
