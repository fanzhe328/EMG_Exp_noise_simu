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
from noise_simulation import proportion_simu

root_path = os.getcwd()
import_module = ("numpy as np", "sklearn.cross_validation",
                 "sklearn.lda", "sklearn.qda", "sklearn.naive_bayes", 
                 "sklearn.svm")


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
        log_fold = root_path + '/result_test1/' + kw['log_fold']
    new_fold(log_fold)

    chan_len = kw['chan_len']
    action_num = kw['action_num']

    cv = 5
    results = []
    results.append(
        ['Feat', 'Algorithm','Channel_Pos', 'Proportion', 'Accuracy', 'std'])
    log_file = 'feat_TD4_intra'

    clf = sklearn.lda.LDA(solver='svd', shrinkage=None, priors=None,
                          n_components=None, store_covariance=False,
                          tol=0.0001)
    # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.8, random_state=random.randrange(1,51))

    # # proportion is 1.0
    # X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(trains, classes, test_size=0.8, random_state=random.randrange(1,51))
    # scores = clf.fit(X_train, y_train).score(X_test, y_test)
    # results.append(['feat_TD4', "LDA(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)", 1, scores.mean(), scores.std()])

    # test_rate_list = [i for i in range(1,9)]
    # test_rate_list = [0.2, 0.4, 0.6, 0.8]
    test_rate_list = [0.2]
    data_num = trains.shape[0]/action_num
    kf = KFold(data_num, n_folds=cv)

    # proportion is 1.0  cv=5
    scores = sklearn.cross_validation.cross_val_score(clf, trains, classes, cv=10)
    results.append(['feat_TD4_cv_'+str(cv), 'lda', 'All', 1.0, scores.mean(), scores.std()])

    for idx, channel_pos in enumerate(kw['pos_list']):
        print '----training TD4 intra, channel_pos: ', channel_pos,'......'

        trains_intra = trains[:,idx*chan_len: idx*chan_len+chan_len]

        iteration = cv
        scores_1_0 = np.zeros((iteration,))
        scores_0_9 = np.zeros((iteration,))
        scores_0_8 = np.zeros((iteration,))
        scores_0_7 = np.zeros((iteration,))
        scores_0_6 = np.zeros((iteration,))
        scores_0_98 = np.zeros((iteration,))
        scores_0_987 = np.zeros((iteration,))
        scores_0_9876 = np.zeros((iteration,))

        iteration -= 1
        for train_idx, test_idx in kf:
            
            # print '------iteration: ', str(5-iteration)
            train_idx_all = np.array([], np.int)
            test_idx_all = np.array([], np.int)
            for action_idx in range(action_num):
                train_idx_all = np.concatenate( (train_idx_all, train_idx*action_idx), axis=0)
                test_idx_all = np.concatenate( (test_idx_all, test_idx*action_idx), axis=0)

            X_train, y_train = trains_intra[train_idx_all], classes[train_idx_all]
            X_test, y_test =  trains_intra[test_idx_all], classes[test_idx_all]
            
            # proportion is 1.0
            scores = clf.fit(X_train, y_train).score(X_test, y_test)
            scores_1_0[iteration] = scores.mean()

            # proportion is 0.9
            trains_noise_9, classes_noise_9 = proportion_simu(
                X_train, y_train, 0.9)
            scores = clf.fit(trains_noise_9, classes_noise_9).score(
                X_test, y_test)
            scores_0_9[iteration] = scores.mean()

            # proportion is 0.8
            trains_noise_8, classes_noise_8 = proportion_simu(
                X_train, y_train, 0.8)
            scores = clf.fit(trains_noise_8, classes_noise_8).score(
                X_test, y_test)
            scores_0_8[iteration] = scores.mean()

            # proportion is 0.7
            trains_noise_7, classes_noise_7 = proportion_simu(
                X_train, y_train, 0.7)
            scores = clf.fit(trains_noise_7, classes_noise_7).score(
                X_test, y_test)
            scores_0_7[iteration] = scores.mean()

            # proportion is 0.6
            trains_noise_6, classes_noise_6 = proportion_simu(
                X_train, y_train, 0.6)
            scores = clf.fit(trains_noise_6, classes_noise_6).score(
                X_test, y_test)
            scores_0_6[iteration] = scores.mean()

            # proportion is 0.8 + 0.9
            trains_noise_98 = np.concatenate(
                (trains_noise_8, trains_noise_9), axis=0)
            classes_noise_98 = np.concatenate(
                (classes_noise_8, classes_noise_9), axis=0)
            scores = clf.fit(trains_noise_98, classes_noise_98).score(
                X_test, y_test)
            scores_0_98[iteration] = scores.mean()

            # proportion is 0.7 + 0.8 + 0.9
            trains_noise_987 = np.concatenate(
                (trains_noise_7, trains_noise_98), axis=0)
            classes_noise_987 = np.concatenate(
                (classes_noise_7, classes_noise_98), axis=0)
            scores = clf.fit(trains_noise_987, classes_noise_987).score(
                X_test, y_test)
            scores_0_987[iteration] = scores.mean()

            # proportion is 0.6 + 0.7 + 0.8 + 0.9
            trains_noise_9876 = np.concatenate(
                (trains_noise_6, trains_noise_987), axis=0)
            classes_noise_9876 = np.concatenate(
                (classes_noise_6, classes_noise_987), axis=0)
            scores = clf.fit(trains_noise_9876, classes_noise_9876).score(
                X_test, y_test)
            scores_0_9876[iteration] = scores.mean()
            


            iteration -= 1
        results.append(['feat_TD4', 'lda', channel_pos, '1.0', np.mean(scores_1_0), np.std(scores_1_0)])
        results.append(['feat_TD4', 'lda', channel_pos, '0.9', np.mean(scores_0_9), np.std(scores_0_9)])
        results.append(['feat_TD4', 'lda', channel_pos, '0.8', np.mean(scores_0_8), np.std(scores_0_8)])
        results.append(['feat_TD4', 'lda', channel_pos, '0.7', np.mean(scores_0_7), np.std(scores_0_7)])
        results.append(['feat_TD4', 'lda', channel_pos, '0.6', np.mean(scores_0_6), np.std(scores_0_6)])
        results.append(['feat_TD4', 'lda', channel_pos, '0.8-0.9', np.mean(scores_0_98), np.std(scores_0_98)])
        results.append(['feat_TD4', 'lda', channel_pos, '0.7-0.9', np.mean(scores_0_987), np.std(scores_0_987)])
        results.append(['feat_TD4', 'lda', channel_pos, '0.6-0.9', np.mean(scores_0_9876), np.std(scores_0_9876)])
        
    log_result(results, log_fold + '/' + log_file + '_action_1-' + str(action_num), 2)
    print '----Log Fold:', log_fold, ', log_file: ', log_file + '_action_1-' + str(action_num)
    print '----training TD4 time elapsed:', time.time() - start_time

def training_lda_TD4_inter(my_clfs, trains, tests, classes, **kw):
    start_time = time.time()
    if(kw.has_key('log_fold')):
        log_fold = root_path + '/result_test1/' + kw['log_fold']
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
            scores_1 = np.zeros((iteration + 1,))
            scores_0_9 = np.zeros((iteration + 1,))
            scores_0_8 = np.zeros((iteration + 1,))
            scores_0_7 = np.zeros((iteration + 1,))
            scores_0_6 = np.zeros((iteration + 1,))
            scores_0_98 = np.zeros((iteration + 1,))
            scores_0_987 = np.zeros((iteration + 1,))
            scores_0_9876 = np.zeros((iteration + 1,))

            while(iteration >= 0):

                # proportion is 1.0
                scores = clf.fit(X_train, y_train).score(X_test, y_test)
                scores_1[iteration] = scores.mean()

                # proportion is 0.9
                trains_noise_9, classes_noise_9 = proportion_simu(
                    X_train, y_train, 0.9)
                scores = clf.fit(trains_noise_9, classes_noise_9).score(
                    X_test, y_test)
                scores_0_9[iteration] = scores.mean()

                # proportion is 0.8
                trains_noise_8, classes_noise_8 = proportion_simu(
                    X_train, y_train, 0.8)
                scores = clf.fit(trains_noise_8, classes_noise_8).score(
                    X_test, y_test)
                scores_0_8[iteration] = scores.mean()

                # proportion is 0.7
                trains_noise_7, classes_noise_7 = proportion_simu(
                    X_train, y_train, 0.7)
                scores = clf.fit(trains_noise_7, classes_noise_7).score(
                    X_test, y_test)
                scores_0_7[iteration] = scores.mean()

                # proportion is 0.6
                trains_noise_6, classes_noise_6 = proportion_simu(
                    X_train, y_train, 0.6)
                scores = clf.fit(trains_noise_6, classes_noise_6).score(
                    X_test, y_test)
                scores_0_6[iteration] = scores.mean()

                # proportion is 0.8 + 0.9
                trains_noise_98 = np.concatenate(
                    (trains_noise_8, trains_noise_9), axis=0)
                classes_noise_98 = np.concatenate(
                    (classes_noise_8, classes_noise_9), axis=0)
                scores = clf.fit(trains_noise_98, classes_noise_98).score(
                    X_test, y_test)
                scores_0_98[iteration] = scores.mean()

                # proportion is 0.7 + 0.8 + 0.9
                trains_noise_987 = np.concatenate(
                    (trains_noise_8, trains_noise_9), axis=0)
                classes_noise_987 = np.concatenate(
                    (classes_noise_8, classes_noise_9), axis=0)
                scores = clf.fit(trains_noise_987, classes_noise_987).score(
                    X_test, y_test)
                scores_0_987[iteration] = scores.mean()

                # proportion is 0.6 + 0.7 + 0.8 + 0.9
                trains_noise_9876 = np.concatenate(
                    (trains_noise_8, trains_noise_9), axis=0)
                classes_noise_9876 = np.concatenate(
                    (classes_noise_8, classes_noise_9), axis=0)
                scores = clf.fit(trains_noise_9876, classes_noise_9876).score(
                    X_test, y_test)
                scores_0_9876[iteration] = scores.mean()

               
                iteration -= 1
            results.append(['feat_TD4', 'lda', channel_pos, '1.0', np.mean(scores_1), np.std(scores_1)])
            results.append(['feat_TD4', 'lda', channel_pos, '0.9', np.mean(scores_0_9), np.std(scores_0_9)])
            results.append(['feat_TD4', 'lda', channel_pos, '0.8', np.mean(scores_0_8), np.std(scores_0_8)])
            results.append(['feat_TD4', 'lda', channel_pos, '0.7', np.mean(scores_0_7), np.std(scores_0_7)])
            results.append(['feat_TD4', 'lda', channel_pos, '0.6', np.mean(scores_0_6), np.std(scores_0_6)])
            results.append(['feat_TD4', 'lda', channel_pos, '0.8-0.9', np.mean(scores_0_98), np.std(scores_0_98)])
            results.append(['feat_TD4', 'lda', channel_pos, '0.7-0.9', np.mean(scores_0_987), np.std(scores_0_987)])
            results.append(['feat_TD4', 'lda', channel_pos, '0.6-0.9', np.mean(scores_0_9876), np.std(scores_0_9876)])

        log_result(results, log_fold + '/' + log_file + '_action_1-' + str(action_num), 2)
        print '----Log Fold:', log_fold, ', log_file: ', log_file + '_action_1-' + str(action_num)
    print '----training TD4 time elapsed:', time.time() - start_time

