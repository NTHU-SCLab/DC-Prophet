import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import svm
import timeit

print("start")

timer_start = timeit.default_timer()

MAX_TIME_INTERVAL = 8353

df = pd.read_csv('/Users/chenhaoyun/Desktop/Y_label/machine_label_Y-500.csv')
X_df = pd.read_csv('/Users/chenhaoyun/Desktop/X_label/machine_label_X-500.csv')

# Generate training data
machine_ID = df['machine ID'].unique()
total_row = None
ID_counter = 0
for ID in machine_ID:
    for time in range(MAX_TIME_INTERVAL + 1 - 6):
        true_time = time + (ID_counter * 8354)
        if true_time == 0:
            total_row = X_df['max CPU usage'].iloc[
                true_time:true_time + 6].values.tolist()
            total_row.extend(X_df['mean CPU usage'].iloc[
                             true_time:true_time + 6].values.tolist())
            total_row.extend(
                X_df['max disk I/O'].iloc[true_time:true_time + 6].values.tolist())
            total_row.extend(
                X_df['mean disk I/O'].iloc[true_time:true_time + 6].values.tolist())
            total_row.extend(X_df['max disk space'].iloc[
                             true_time:true_time + 6].values.tolist())
            total_row.extend(X_df['mean disk space'].iloc[
                             true_time:true_time + 6].values.tolist())
            total_row.extend(X_df['max memory usage'].iloc[
                             true_time:true_time + 6].values.tolist())
            total_row.extend(X_df['mean memory usage'].iloc[
                             true_time:true_time + 6].values.tolist())
            total_row.extend(X_df['max page cache'].iloc[
                             true_time:true_time + 6].values.tolist())
            total_row.extend(X_df['mean page cache'].iloc[
                             true_time:true_time + 6].values.tolist())
            total_row.extend(X_df['max MAI'].iloc[
                             true_time:true_time + 6].values.tolist())
            total_row.extend(X_df['mean MAI'].iloc[
                             true_time:true_time + 6].values.tolist())
            total_row.append(df['Y label'].iloc[true_time + 6])
        else:
            tmp_row = X_df['max CPU usage'].iloc[
                true_time:true_time + 6].values.tolist()
            tmp_row.extend(X_df['mean CPU usage'].iloc[
                           true_time:true_time + 6].values.tolist())
            tmp_row.extend(
                X_df['max disk I/O'].iloc[true_time:true_time + 6].values.tolist())
            tmp_row.extend(
                X_df['mean disk I/O'].iloc[true_time:true_time + 6].values.tolist())
            tmp_row.extend(X_df['max disk space'].iloc[
                           true_time:true_time + 6].values.tolist())
            tmp_row.extend(X_df['mean disk space'].iloc[
                           true_time:true_time + 6].values.tolist())
            tmp_row.extend(X_df['max memory usage'].iloc[
                           true_time:true_time + 6].values.tolist())
            tmp_row.extend(X_df['mean memory usage'].iloc[
                           true_time:true_time + 6].values.tolist())
            tmp_row.extend(X_df['max page cache'].iloc[
                           true_time:true_time + 6].values.tolist())
            tmp_row.extend(X_df['mean page cache'].iloc[
                           true_time:true_time + 6].values.tolist())
            tmp_row.extend(X_df['max MAI'].iloc[
                           true_time:true_time + 6].values.tolist())
            tmp_row.extend(X_df['mean MAI'].iloc[
                           true_time:true_time + 6].values.tolist())
            tmp_row.append(df['Y label'].iloc[true_time + 6])
            total_row.extend(tmp_row)
            del tmp_row
    ID_counter += 1

total_row = np.array(total_row)
total_row = total_row.reshape(-1, 73)
training_df = pd.DataFrame(total_row, columns=['max CPU usage 1', 'max CPU usage 2',
                                               'max CPU usage 3', 'max CPU usage 4',
                                               'max CPU usage 5', 'max CPU usage 6',
                                               'mean CPU usage 1', 'mean CPU usage 2',
                                               'mean CPU usage 3', 'mean CPU usage 4',
                                               'mean CPU usage 5', 'mean CPU usage 6',
                                               'max disk I/O 1', 'max disk I/O 2',
                                               'max disk I/O 3', 'max disk I/O 4',
                                               'max disk I/O 5', 'max disk I/O 6',
                                               'mean disk I/O 1', 'mean disk I/O 2',
                                               'mean disk I/O 3', 'mean disk I/O 4',
                                               'mean disk I/O 5', 'mean disk I/O 6',
                                               'max disk space 1', 'max disk space 2',
                                               'max disk space 3', 'max disk space 4',
                                               'max disk space 5', 'max disk space 6',
                                               'mean disk space 1', 'mean disk space 2',
                                               'mean disk space 3', 'mean disk space 4',
                                               'mean disk space 5', 'mean disk space 6',
                                               'max memory usage 1', 'max memory usage 2',
                                               'max memory usage 3', 'max memory usage 4',
                                               'max memory usage 5', 'max memory usage 6',
                                               'mean memory usage 1', 'mean memory usage 2',
                                               'mean memory usage 3', 'mean memory usage 4',
                                               'mean memory usage 5', 'mean memory usage 6',
                                               'max page cache 1', 'max page cache 2',
                                               'max page cache 3', 'max page cache 4',
                                               'max page cache 5', 'max page cache 6',
                                               'mean page cache 1', 'mean page cache 2',
                                               'mean page cache 3', 'mean page cache 4',
                                               'mean page cache 5', 'mean page cache 6',
                                               'max MAI 1', 'max MAI 2',
                                               'max MAI 3', 'max MAI 4',
                                               'max MAI 5', 'max MAI 6',
                                               'mean MAI 1', 'mean MAI 2',
                                               'mean MAI 3', 'mean MAI 4',
                                               'mean MAI 5', 'mean MAI 6',
                                               'Y_label'])

# drop off ['Y_label'] == -1
training_df = training_df[training_df['Y_label'] != -1]

training_df.to_csv(
    '/Users/chenhaoyun/Desktop/machine_label_Training-500.csv', encoding='utf-8', index=False)

timer_end = timeit.default_timer()
print(timer_end - timer_start)
# @ timeit single .csv generate features : 4206s


X = training_df

# One-class SVM is an unsupervised algorithm that learns a decision function for novelty detection:
# classifying new data as similar or different to the training set.

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)  # Define the split - into 5 folds
# returns the number of splitting iterations in the cross-validator
kf.get_n_splits(X)
print(kf)

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid

# via grid-search to find the best hyperparameter
best_hyperparameter = [0.0, 0.0]  # [nu, gamma]
f3_score_best = 0.0
#param_grid = {'nus' : np.linspace(0.01, 0.99, 99), 'gammas': np.logspace(-9, 3, 13)}
param_grid = {'nus': np.linspace(
    0.01, 0.01, 1), 'gammas': np.logspace(-9, -9, 1)}  # for test
grid = ParameterGrid(param_grid)
for params in grid:
    f3_score_total = 0.0
    for train_index, test_index in kf.split(X):
        print('TRAIN:', train_index, 'TEST:', test_index)
        tmp = X.iloc[train_index]
        X_train = tmp[tmp['Y_label'] == 0].drop(['Y_label'], 1)
        X_test = X.drop(['Y_label'], 1).iloc[test_index]
        Y_test_tmp = X['Y_label'].iloc[test_index]
        Y_test = np.where(Y_test_tmp == 0, 1, -1)
        clf = svm.OneClassSVM(
            nu=params['nus'], kernel="rbf", gamma=params['gammas'])
        clf.fit(X_train)
        X_pred = clf.predict(X_test)
        print(X_pred)
        print(Y_test)

        from sklearn.metrics import accuracy_score
        from sklearn.metrics import classification_report
        print(accuracy_score(Y_test, X_pred, normalize=False))
        print(classification_report(Y_test, X_pred))

        precision, recall, f3_score, support = precision_recall_fscore_support(
            Y_test, X_pred, beta=3.0)
        f3_score_total += f3_score[1]

    print("f3_score = {}".format(f3_score_total / 5))
    if f3_score_total / 5 > f3_score_best:
        f3_score_best = f3_score_total / 5
        best_hyperparameter[0], best_hyperparameter[
            1] = params['nus'], params['gammas']

print("\n\nf3_score_best = {}".format(f3_score_best))
print("best hyperparameter nu = {}".format(best_hyperparameter[0]))
print("best hyperparameter gamma = {}".format(best_hyperparameter[1]))
# drop off ['Y_label'] == -1
training_df = training_df[training_df['Y_label'] != -1]

# @ timeit single .csv generate features : 677s
