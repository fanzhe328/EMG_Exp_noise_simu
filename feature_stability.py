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

def feature_stability_O2O(channel_group='1', feature_type='TD4', feat_num=4):
    ''' 对每个特征，分析其在不同移位程度的的均值和方差 '''
    channel_pos_list = ['S0',                                             # 中心位置
                        'U1', 'U2', 'D1', 'D2', 'L1', 'L2', 'R1', 'R2']  # 上 下 左 右
    subjects = ['subject_' + str(i + 1) for i in range(2)]
    # print subjects
    # sys.exit(0)
    for subject in subjects:
        title_pre = subject 
        actions = [i + 1 for i in range(6)]
        # actions = [1]
        for action in actions:
            print 'Generate action: ', action
            filename = title_pre + '_feat_' + feature_type + '_action_' + str(action)
            title = title_pre + '_action_' + str(action)
            feature = np.load(root_path + '/train4_250_100/' + filename + '.npy')

            channel_num = 4                                     # 四通道
            channel_span = feat_num*channel_num                 # 双通道中一组电极两个位置的跨度

            feature_list = ['MAV', 'ZC', 'SSC', 'WL']

            for feat_ind, feature_name in enumerate(feature_list):

                means_list = np.ones((channel_num/channel_group_len,))
                stds_list = np.ones((channel_num/channel_group_len,))
                # print type(means_list[0]), means_list.shape
                # print feature.shape
                for i in xrange(channel_num/channel_group_len):
                    # print i*feat_num+feat_ind, i*feat_num+feat_ind+channel_span*feat_num

                    if channel_group == '1&2':
                        # group 1 and 2
                        temp = np.concatenate(
                            (feature[:, i*feat_num+feat_ind], feature[:, i*feat_num+feat_ind+channel_span*feat_num]),
                            axis = None)
                    elif channel_group == '1':
                        # group 1
                        temp = feature[:, i*feat_num+feat_ind]
                    elif channel_group == '2':
                        # group 2
                        temp = feature[:, i*feat_num+feat_ind+channel_span*feat_num]

                    means_list[i] = np.mean(temp, axis=0)
                    stds_list[i] = np.std(temp, axis=0)

                labels = np.arange(channel_num)
                ind = np.array([i * 2 for i in range(channel_num/channel_group_len)])
                width = 0.8

                # plt.figure(num=1, figsize=(8,6))
                fig, ax = plt.subplots(figsize=(16, 6))
                ax.bar(ind, means_list, width, color='r', yerr=stds_list)

                ax.set_ylabel('Scores')
                # ax.set_xlim(0,64)
                ax.set_title(title + '_group_1&2_O2O_' + feature_name)
                ax.set_xticks(ind + width / 2)
                ax.set_xticklabels(channel_pos_list)
                # plt.show()
                # if feat_ind == 1:
                #     sys.exit(0)
                plt.savefig('result/figure/stability/' +
                            title + '_group_'+channel_group+'_O2O_' + feature_name, dpi=120)
                plt.close()

def feature_stability_O2A(channel_group='1', feature_type='TD4', feat_num=4):
    ''' 对每个特征，分析其在不移位和在所有情况下的均值和方差 '''
    channel_pos_list = ['S0',                                             # 中心位置
                    'U1', 'U2', 'D1', 'D2', 'L1', 'L2', 'R1', 'R2']  # 上 下 左 右
    
    subjects = ['subject_' + str(i + 1) for i in range(1)]
    actions = [i + 1 for i in range(1)]

    feature_list = ['MAV', 'ZC', 'SSC', 'WL']
    xtickslabels_post = ['O', 'A']
    xtickslabels = [x+'_'+y for x in feature_list for y in xtickslabels_post]
    for subject in subjects: 
        for action in actions:
            # print 'Generate action: ', action
            
            filename = subject + '_feat_' + feature_type + '_action_' + str(action)

            feature = np.load(root_path + '/train4_250_100/' + filename + '.npy')
            chan_num = 4                    # 四通道
            channel_span = 2*feat_num               # 四通道中一组电极两个位置的跨度

            data_len = channel_span
            means_list = np.ones((data_len,))
            stds_list = np.ones((data_len,))

            # print feature.shape
            for feat_ind, feature_name in enumerate(feature_list):
                print 'feature_name: ', feature_name, 'channel_group: ', channel_group
                # 正常位置的特征数据
                idx = []
                if channel_group == 0:
                    # group all
                    idx = [feat_ind + i*feat_num for i in range(4)]
                    feature_O = feature[:, idx]

                else:
                    # group 1 2 3 4
                    idx = feat_ind + (channel_group-1)*feat_num
                    feature_O = feature[:, feat_ind + (channel_group-1)*feat_num]
                    # print feat_ind + (channel_group-1)*feat_num
                feature_O = feature_O.reshape((-1,1))
                # 所有移位情况下的数据
                feature_A = np.array([])
                idx = []
                for i in xrange(len(channel_pos_list)-1):
                    # print i*feat_num+feat_ind, i*feat_num+feat_ind+channel_span*feat_num

                    if channel_group == 0:
                        # group all
                        idx_temp = [(i+1)*channel_span + j*chan_num + feat_ind for j in range(4)]
                        idx.extend(idx_temp)
                    else:
                        # group 1 2 3 4
                        idx_temp = (i+1)*channel_span+(channel_group-1)*feat_num+feat_ind
                        idx.append(idx_temp)
                        # print (i+1)*channel_span+(channel_group-1)*feat_num+feat_ind
                print 'idx_s:' ,idx
                feature_A = feature[:, idx].reshape((-1, 1))
                
                means_list[feat_ind*2] = np.mean(feature_O, axis=0)
                stds_list[feat_ind*2] = np.std(feature_O, axis=0)
                means_list[feat_ind*2+1] = np.mean(feature_A, axis=0)
                stds_list[feat_ind*2+1] = np.std(feature_A, axis=0)

                bar_num = 2
                labels = np.arange(bar_num)
                ind = np.array([i * 1 for i in range(bar_num)])
                width = 0.3

                # plt.figure(num=1, figsize=(8,6))
                fig, ax = plt.subplots(figsize=(8,6))
                ax.bar(ind, means_list[feat_ind*bar_num:feat_ind*bar_num+bar_num], width, 
                    color='r', yerr=stds_list[feat_ind*bar_num:feat_ind*bar_num+bar_num])

                ax.set_ylabel('Scores')
                # ax.set_xlim(0,64)
                ax.set_title(subject + '_action_' + str(action) + '_group_' + str(channel_group) + '_O2A_' + feature_name)
                ax.set_xticks(ind + width / 2)
                ax.set_xticklabels(xtickslabels_post)
                
                fold_path = 'result/figure/stability/' 
                new_fold(fold_path)
                plt.savefig(fold_path +
                            subject + '_action_' + str(action) +  '_group_' + str(channel_group) + '_O2A_' 
                            + feature_name, dpi=120)
                plt.close()
            
            bar_num = data_len
            print means_list
            labels = np.arange(bar_num)
            ind = np.array([i * 1 for i in range(bar_num)])
            width = 0.7

            # plt.figure(num=1, figsize=(8,6))
            fig, ax = plt.subplots(figsize=(8,6))
            ax.bar(ind, means_list, width, color='r', yerr=stds_list)

            ax.set_ylabel('Scores')
            ax.set_xlim(0, 8)
            ax.set_title( subject + '_action_' + str(action) + '_group_' + str(channel_group) + '_O2A')
            ax.set_xticks(ind+width/2)
            ax.set_xticklabels(xtickslabels)
            # plt.show()
            plt.savefig('result/figure/stability/' +
                        subject + '_action_' + str(action) + '_group_' + str(channel_group) + '_O2A', dpi=120)
            plt.close()

if __name__ == '__main__':
    group_list = [i for i in range(0,5)]
    # print group_list
    # group_list = ['1']
    for group in group_list:
        # feature_stability_O2O(group)
        feature_stability_O2A(group)

