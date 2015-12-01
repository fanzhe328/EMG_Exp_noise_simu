# *-* coding: UTF-8 *-*
import os, sys, re, time
import numpy as np
from file_script import log_result, new_fold

root_path = os.getcwd()


# def intra_result_combination(
#         fold_pre='result', win_inc='250_100', fold='TD4_data4_subject_1_norm',
#         file_pre='feat_TD4_', file_post='1'):
#     res_fold = root_path+'/'+fold_pre+'/'+win_inc+'/'+fold
#     dis = os.listdir(res_fold)
#     os.chdir(res_fold)
    
#     diss = []
#     for di in dis:
#     	if re.match('feat_TD4_(\w+)_1\.npy', di):
#     		diss.append(di)

#     # print res_fold
#     ress = np.array([])
#     ress_ylime = 0
#     start_line = 0
#     for di in diss:
#     	data = np.load(di)
#     	if start_line == 0:
#     		ress = np.concatenate( (ress, data), axis=None)
#     		start_line = 1
#     	elif start_line == 1:
#     		ress = np.concatenate( (ress, data[start_line:,:]), axis=None)
#     	if ress_ylime == 0:
#     		ress_ylime = data.shape[1]
#     	print data.shape, ress.shape
#     ress = ress.reshape( (-1, ress_ylime))
#     # print ress.shape
#     log_result(ress, 'feat_TD4_intra.npy', 2)

def result_load(dir='250_100', feature_type='TD4', subject='subject_1', norm='_norm', action='7', training_type='intra'):
    file_path = root_path + '/result/' + dir + '/' +\
                feature_type+'_data4_'+ subject + norm + '/' +\
                'feat_'+feature_type+'_'+training_type+'_action_1-'+str(action)+'.npy'
    data = np.load(file_path)
    return data

def result_integrate_intra(time_now):
    fold_path = root_path + '/result/proportional_integrate'
    new_fold(fold_path) 
    feature_type = 'TD4'
    norm = '_norm'
    training_type = 'intra'
    channel_pos_list = ['S0',                                             # 中心位置
                    'U1', 'U2', 'D1', 'D2', 'L1', 'L2', 'R1', 'R2']  # 上 下 左 右
    proportional_list = ['1.0', '0.9', '0.8', '0.8+0.9', '1.1', '1.2', '1.1+1.2', '1.1+0.9', 'All']
    action_lists = [7, 9, 11]
    subject_list = ['subject_' + str(i) for i in range(1, 6)]

    res_all = []
    blank_line = ['','','','','','','','','','']
    res_all.append(blank_line)

    for action in action_lists:
        for subject in subject_list:
            res = []
            index = 2
            span = 9
            res_ind = 1

            data = result_load('250_100',feature_type, subject, norm, action, training_type)
            title = feature_type+'_'+subject+'_action_1-'+str(action)
            res_head = [title]
            res_head.extend(proportional_list)
            res.append(res_head)
            
            for i in range(len(channel_pos_list)-1):
                res_intra = [channel_pos_list[i+1]]
                res_intra.extend(map(float,data[index:index+span,4][:]))
                index += span
                res.append(res_intra)

            res_np = np.array(res)
            res_aver = ['average']
            for i in range(len(proportional_list)):
                res_aver.append(np.mean(map(float,res_np[res_ind:res_ind+8,i+1])))
            res.append(res_aver)
            file_path = fold_path + '/prop_intra_'+title+'_'+str(time_now)
            log_result(res, file_path, 2)

            res_all.extend(res)
            res_all.append(blank_line)

    file_path = fold_path + '/prop_intra_all_'+str(time_now)
    log_result(res_all, file_path, 2)

if __name__ == '__main__':
    time_now = time.strftime('%Y-%m-%d_%H-%M',time.localtime(time.time()))
    # print time_now
    result_integrate_intra(time_now)
