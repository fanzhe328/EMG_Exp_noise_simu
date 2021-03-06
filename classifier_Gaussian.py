# *-* coding=utf-8 *-*
#!/usr/lib/python
import numpy as np
import os
import time
import sys
import random
import sklearn
import sklearn.lda
import sklearn.qda
import sklearn.naive_bayes
import sklearn.cross_validation
import sklearn.svm

from sklearn.cross_validation import KFold
from noise_simulation import gaussian_simu, proportion_simu

root_path = os.getcwd()



def log_result(results, log_file, flag):
    np.save(log_file + '.npy', results)
    if flag == 2:
        np.savetxt(log_file + '.csv', results, fmt="%s", delimiter=",")


def new_fold(log_fold):
    if os.path.isdir(log_fold) == False:
        try:
            os.makedirs(log_fold)
        except:
            print "Can not create log fold! "
            return False
    return True


def training_lda_TD4_intra(my_clfs, trains, classes, **kw):
    start_time = time.time()
    if(kw.has_key('log_fold')):
        log_fold = root_path + '/result_gaussian/' + kw['log_fold']
    new_fold(log_fold)

    chan_len = kw['chan_len']
    chan_num = kw['chan_num']
    action_num = kw['action_num']
    subject = kw['subject']
    feat_num = kw['feat_num']

    cv = 5
    results = []
    results.append(
        ['Feat', 'Algorithm','Channel_Pos', 'Gaussian_simu', 'Accuracy', 'std'])
    log_file = 'feat_TD4_intra'

    clf = sklearn.lda.LDA(solver='svd', shrinkage=None, priors=None,
                          n_components=None, store_covariance=False,
                          tol=0.0001)
    
    data_num = trains.shape[0]/action_num
    kf = KFold(data_num, n_folds=cv)

    # proportion is 1.0  cv=5
    scores = sklearn.cross_validation.cross_val_score(clf, trains, classes, cv=10)
    results.append(['feat_TD4_cv_'+str(cv), 'lda', 'All', 1.0, scores.mean(), scores.std()])

    for idx, channel_pos in enumerate(kw['pos_list']):
        if channel_pos=='S0':
            continue
        print '----training TD4 intra, channel_pos: ', channel_pos,'......'

        trains_intra = trains[:,idx*chan_len: idx*chan_len+chan_len]

        iteration = cv
        scores = np.zeros((iteration,))
        scores_Gaussian = np.zeros((iteration,))
        iteration -= 1
        for train_idx, test_idx in kf:
            
            train_idx_all = np.array([], np.int)
            test_idx_all = np.array([], np.int)
            for action_idx in range(action_num):
                train_idx_all = np.concatenate( (train_idx_all, train_idx*action_idx), axis=0)
                test_idx_all = np.concatenate( (test_idx_all, test_idx*action_idx), axis=0)

            # base
            X_train, y_train = trains_intra[train_idx_all], classes[train_idx_all]
            X_test, y_test =  trains_intra[test_idx_all], classes[test_idx_all]
            score = clf.fit(X_train, y_train).score(X_test, y_test)
            scores[iteration] = score.mean()

            # Gaussian simu
            X_train, y_train = gaussian_simu(trains_intra[train_idx_all], classes[train_idx_all], subject, action_num, chan_num, feat_num)
            X_test, y_test =  trains_intra[test_idx_all], classes[test_idx_all]
            # print X_train
            # print y_train
            # print classes[train_idx_all]
            # sys.exit(0)
            score = clf.fit(X_train, y_train).score(X_test, y_test)
            scores_Gaussian[iteration] = score.mean()


            iteration -= 1
        results.append(['feat_TD4', 'lda', channel_pos, 'False', np.mean(scores), np.std(scores)])
        results.append(['feat_TD4', 'lda', channel_pos, 'True', np.mean(scores_Gaussian), np.std(scores_Gaussian)])

    log_result(results, log_fold + '/' + log_file + '_action_1-' + str(action_num), 2)
    print '----Log Fold:', log_fold, ', log_file: ', log_file + '_action_1-' + str(action_num)
    print '----training TD4 time elapsed:', time.time() - start_time

def training_lda_TD4_inter(my_clfs, trains, tests, classes, **kw):
    start_time = time.time()
    if(kw.has_key('log_fold')):
        log_fold = root_path + '/result/' + kw['log_fold']
    new_fold(log_fold)

    chan_len = kw['chan_len']
    action_num = kw['action_num']

    print "----training TD4 inter, training by position O, testing by electrode shift ", 

    cv = 5
    results = []
    results.append(
        ['Feat', 'Algorithm','Channel_Pos', 'Proportion', 'Accuracy', 'std'])
    log_file = 'feat_TD4_inter'

    clf = sklearn.lda.LDA(solver='svd', shrinkage=None, priors=None,
                          n_components=None, store_covariance=False,
                          tol=0.0001)
    test_rate_list = [0.2]

    scores = sklearn.cross_validation.cross_val_score(
        clf, trains, classes, cv=10)
    results.append(['feat_TD4_cv_5', 'lda', 'O',
                    1.0, scores.mean(), scores.std()])

    for i in test_rate_list:
        for idx, channel_pos in enumerate(kw['pos_list']):

            X_train = trains
            y_train = classes
            X_test = tests[:,idx*chan_len:idx*chan_len+chan_len]
            y_test = classes
            # print tests.shape, chan_len+chan_len
            # print X_train.shape, y_train.shape, X_test.shape
            # sys.exit(0)

            iteration = 1
            scores_1_0 = np.zeros((iteration + 1,))
            scores_1_1 = np.zeros((iteration + 1,))
            scores_1_2 = np.zeros((iteration + 1,))
            scores_1_12 = np.zeros((iteration + 1,))
            scores_0_9 = np.zeros((iteration + 1,))
            scores_0_8 = np.zeros((iteration + 1,))
            scores_0_89 = np.zeros((iteration + 1,))
            scores_19 = np.zeros((iteration + 1,))
            scores_1289 = np.zeros((iteration + 1,))

            while(iteration >= 0):
                # print 'iteration: ', str(1-iteration), ', test_size:', i
                # print X_train.shape, X_test.shape, y_train.shape, y_test.shape

                # proportion is 1.0
                scores = clf.fit(X_train, y_train).score(X_test, y_test)
                scores_1_0[iteration] = scores.mean()
                # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', 1.0, scores.mean(), scores.std()])

                # proportion is 1.1
                # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
                trains_noise_1, classes_noise_1 = proportion_simu(
                    X_train, y_train, 1.1)
                scores = clf.fit(trains_noise_1, classes_noise_1).score(
                    X_test, y_test)
                scores_1_1[iteration] = scores.mean()

                # proportion is 1.2
                # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
                trains_noise_2, classes_noise_2 = proportion_simu(
                    X_train, y_train, 1.2)
                scores = clf.fit(trains_noise_2, classes_noise_2).score(
                    X_test, y_test)
                scores_1_2[iteration] = scores.mean()

                # # proportion is 1.1 + 1.2
                trains_noise_12 = np.concatenate(
                    (trains_noise_1, trains_noise_2), axis=0)
                classes_noise_12 = np.concatenate(
                    (classes_noise_1, classes_noise_2), axis=0)
                scores = clf.fit(trains_noise_12, classes_noise_12).score(
                    X_test, y_test)
                scores_1_12[iteration] = scores.mean()

                # proportion is 0.9
                # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
                trains_noise_9, classes_noise_9 = proportion_simu(
                    X_train, y_train, 0.9)
                scores = clf.fit(trains_noise_9, classes_noise_9).score(
                    X_test, y_test)
                scores_0_9[iteration] = scores.mean()
                # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.9', scores.mean(), scores.std()])

                # proportion is 0.8
                # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
                trains_noise_8, classes_noise_8 = proportion_simu(
                    X_train, y_train, 0.8)
                scores = clf.fit(trains_noise_8, classes_noise_8).score(
                    X_test, y_test)
                scores_0_8[iteration] = scores.mean()
                # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.8', scores.mean(), scores.std()])

                # # proportion is 0.8 + 0.9
                trains_noise_89 = np.concatenate(
                    (trains_noise_8, trains_noise_9), axis=0)
                classes_noise_89 = np.concatenate(
                    (classes_noise_8, classes_noise_9), axis=0)
                scores = clf.fit(trains_noise_89, classes_noise_89).score(
                    X_test, y_test)
                scores_0_89[iteration] = scores.mean()
                # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.8+0.9', scores.mean(), scores.std()])

                # # proportion is 1.1 + 0.9
                trains_noise_19 = np.concatenate(
                    (trains_noise_1, trains_noise_9), axis=0)
                classes_noise_19 = np.concatenate(
                    (classes_noise_1, classes_noise_9), axis=0)
                scores = clf.fit(trains_noise_19, classes_noise_19).score(
                    X_test, y_test)
                scores_19[iteration] = scores.mean()

                # # proportion is 1.1 + 0.9 + 1.2 + 0.8
                trains_noise_1289 = np.concatenate(
                    (trains_noise_12, trains_noise_89), axis=0)
                classes_noise_1289 = np.concatenate(
                    (classes_noise_12, classes_noise_89), axis=0)
                scores = clf.fit(trains_noise_1289,
                                 classes_noise_1289).score(X_test, y_test)
                scores_1289[iteration] = scores.mean()

                iteration -= 1
            results.append(['feat_TD4', 'lda', channel_pos, '1.0', np.mean(scores_1_0), np.std(scores_1_0)])
            results.append(['feat_TD4', 'lda', channel_pos, '0.9', np.mean(scores_0_9), np.std(scores_0_9)])
            results.append(['feat_TD4', 'lda', channel_pos, '0.8', np.mean(scores_0_8), np.std(scores_0_8)])
            results.append(['feat_TD4', 'lda', channel_pos, '0.8+0.9', np.mean(scores_0_89), np.std(scores_0_89)])
            results.append(['feat_TD4', 'lda', channel_pos, '1.1', np.mean(scores_1_1), np.std(scores_1_1)])
            results.append(['feat_TD4', 'lda', channel_pos, '1.2', np.mean(scores_1_2), np.std(scores_1_2)])
            results.append(['feat_TD4', 'lda', channel_pos, '1.1+1.2', np.mean(scores_1_12), np.std(scores_1_12)])
            results.append(['feat_TD4', 'lda', channel_pos, '1.1+0.9', np.mean(scores_19), np.std(scores_19)])
            results.append(['feat_TD4', 'lda', channel_pos, '1.1+0.9+1.2+0.8', np.mean(scores_1289), np.std(scores_1289)])

        log_result(results, log_fold + '/' + log_file + '_action_1-' + str(action_num), 2)
        print '----Log Fold:', log_fold, ', log_file: ', log_file + '_action_1-' + str(action_num)
    print '----training TD4 time elapsed:', time.time() - start_time



# def training_lda_TD4_cross(my_clfs, X_train, y_train, X_test, y_test, **kw):
#     start_time = time.time()
#     print "training TD4 cross............."

#     if(kw.has_key('log_fold')):
#         log_fold = root_path + '/result/' + kw['log_fold']
#     if os.path.isdir(log_fold) == False:
#         try:
#             os.makedirs(log_fold)
#         except:
#             print "Can not create log fold! "
#             return
#     os.chdir(log_fold)

#     results = []
#     results.append(['Feat', 'Algorithm', 'Proportion', 'Accuracy', 'std'])
#     log_file = 'feat_TD4'
#     clf = sklearn.lda.LDA(solver='svd', shrinkage=None, priors=None,
#                           n_components=None, store_covariance=False, tol=0.0001)
#     test_rate_list = [0.2, 0.4, 0.6, 0.8]

#     for i in test_rate_list:

#         iteration = 4

#         scores_1_0 = np.zeros((iteration + 1,))
#         scores_0_9 = np.zeros((iteration + 1,))
#         scores_0_8 = np.zeros((iteration + 1,))
#         scores_0_89 = np.zeros((iteration + 1,))

#         while(iteration >= 0):
#             print 'test_rate:', i, 'iteration:', iteration
#             # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=i, random_state=random.randrange(1,51))
#             # print 'test_size:', i
#             print X_train.shape, X_test.shape, y_train.shape, y_test.shape

#             # proportion is 1.0
#             scores = clf.fit(X_train, y_train).score(X_test, y_test)
#             scores_1_0[iteration] = scores.mean()
#             # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', 1.0, scores.mean(), scores.std()])

#             # proportion is 0.9
#             # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
#             trains_noise_1, classes_noise_1 = proportion_simu(
#                 X_train, y_train, 0.9)
#             scores = clf.fit(trains_noise_1, classes_noise_1).score(
#                 X_test, y_test)
#             scores_0_9[iteration] = scores.mean()
#             # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.9', scores.mean(), scores.std()])

#             # proportion is 0.8
#             # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.2, random_state=random.randrange(1,51))
#             trains_noise_2, classes_noise_2 = proportion_simu(
#                 X_train, y_train, 0.8)

#             scores = clf.fit(trains_noise_2, classes_noise_2).score(
#                 X_test, y_test)
#             scores_0_8[iteration] = scores.mean()
#             # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.8', scores.mean(), scores.std()])

#             # # proportion is 0.8 + 0.9
#             trains_noise_3 = np.concatenate(
#                 (trains_noise_1, trains_noise_2), axis=0)
#             classes_noise_3 = np.concatenate(
#                 (classes_noise_1, classes_noise_2), axis=0)
#             scores = clf.fit(trains_noise_3, classes_noise_3).score(
#                 X_test, y_test)
#             scores_0_89[iteration] = scores.mean()
#             # results.append(['feat_TD4', 'lda(svd,tol=0.0001)', '0.8+0.9', scores.mean(), scores.std()])

#             iteration -= 1
#         results.append(['feat_TD4', 'lda(svd;tol=0.0001;test_rate=' +
#                         str(i) + ')', '1.0', np.mean(scores_1_0), np.std(scores_1_0)])
#         results.append(['feat_TD4', 'lda(svd;tol=0.0001;test_rate=' +
#                         str(i) + ')', '0.9', np.mean(scores_0_9), np.std(scores_0_9)])
#         results.append(['feat_TD4', 'lda(svd;tol=0.0001;test_rate=' +
#                         str(i) + ')', '0.8', np.mean(scores_0_8), np.std(scores_0_8)])
#         results.append(['feat_TD4', 'lda(svd;tol=0.0001;test_rate=' + str(i) + ')',
#                         '0.8+0.9', np.mean(scores_0_89), np.std(scores_0_89)])

#     log_result(results, log_file + '_' + str(kw['num']), 2)
#     print 'training TD4 time elapsed:', time.time() - start_time