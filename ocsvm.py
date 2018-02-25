import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import svm

MAX_TIME_INTERVAL = 8353

df = pd.read_csv('/Users/chenhaoyun/Desktop/Y_label/machine_label_Y-500.csv')

df['average'] = np.random.randint(100, size=4177000)
df['peak'] = np.random.randint(100, size=4177000)

# Generate training data
machine_ID = df['machine ID'].unique()
training_df = None
create_training_df = False
ID_counter = 0
for ID in machine_ID:
    print(ID)
    for time in range(MAX_TIME_INTERVAL + 1 - 6):
        true_time = time + (ID_counter * 8354)
        if create_training_df == False:
            training_df = df['average'].iloc[true_time:true_time + 6].values
            training_df = np.append(training_df, df['peak'].iloc[
                                    true_time:true_time + 6].values)
            training_df = np.append(
                training_df, df['Y label'].iloc[true_time + 6])
            training_df = pd.DataFrame(training_df.reshape(-1, len(training_df)), columns=['avg1', 'avg2',
                                                                                           'avg3', 'avg4',
                                                                                           'avg5', 'avg6',
                                                                                           'peak1', 'peak2',
                                                                                           'peak3', 'peak4',
                                                                                           'peak5', 'peak6', 'Y_label'])
            create_training_df = True
        else:
            tmp_training_df = df['average'].iloc[
                true_time:true_time + 6].values
            tmp_training_df = np.append(tmp_training_df, df['peak'].iloc[
                                        true_time:true_time + 6].values)
            tmp_training_df = np.append(
                tmp_training_df, df['Y label'].iloc[true_time + 6])
            tmp_training_df = pd.DataFrame(tmp_training_df.reshape(-1, len(tmp_training_df)), columns=['avg1', 'avg2',
                                                                                                       'avg3', 'avg4',
                                                                                                       'avg5', 'avg6',
                                                                                                       'peak1', 'peak2',
                                                                                                       'peak3', 'peak4',
                                                                                                       'peak5', 'peak6', 'Y_label'])
            frames = [training_df, tmp_training_df]
            training_df = pd.concat(frames, ignore_index=True)

    # drop off ['Y_label'] == -1
    training_df = training_df[training_df['Y_label'] != -1]
    ID_counter += 1
    print(training_df)
    break  # for test only get single machine ID == 5

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
        print(classification_report(y_true, y_pred, target_names=target_names))

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
