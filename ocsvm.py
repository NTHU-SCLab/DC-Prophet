import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import svm

print("start")
MAX_TIME_INTERVAL = 8353

df = pd.read_csv('/Users/chenhaoyun/Desktop/Y_label/machine_label_Y-500.csv')

# Random X features to test one-class svm
df['average'] = np.random.randint(100, size=4177000)
df['peak'] = np.random.randint(100, size=4177000)

# Generate training data
machine_ID = df['machine ID'].unique()
total_row = None
ID_counter = 0
for ID in machine_ID:
    for time in range(MAX_TIME_INTERVAL + 1 - 6):
        true_time = time + (ID_counter * 8354)
        if true_time == 0:
            total_row = df['average'].iloc[
                true_time:true_time + 6].values.tolist()
            total_row.extend(
                df['peak'].iloc[true_time:true_time + 6].values.tolist())
            total_row.append(df['Y label'].iloc[true_time + 6])
        else:
            tmp_row = df['average'].iloc[
                true_time:true_time + 6].values.tolist()
            tmp_row.extend(df['peak'].iloc[
                           true_time:true_time + 6].values.tolist())
            tmp_row.append(df['Y label'].iloc[true_time + 6])
            total_row.extend(tmp_row)
            del tmp_row
    ID_counter += 1

total_row = np.array(total_row)
total_row = total_row.reshape(-1, 13)
training_df = pd.DataFrame(total_row, columns=['avg1', 'avg2',
                                               'avg3', 'avg4',
                                               'avg25', 'avg6',
                                               'peak1', 'peak2',
                                               'peak3', 'peak4',
                                               'peak5', 'peak6', 'Y_label'])


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
