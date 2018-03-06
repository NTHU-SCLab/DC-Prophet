# temp test for random forest
print(training_df[training_df['Y_label'] == 1].shape)
print(training_df[training_df['Y_label'] == 2].shape)
print(training_df[training_df['Y_label'] == 3].shape)

abnormal_training_df = training_df[training_df['Y_label'] >= 1]
X = abnormal_training_df
###

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)  # Define the split - into 5 folds
# returns the number of splitting iterations in the cross-validator
kf.get_n_splits(X)
# print(kf)

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid

# via grid-search to find the best hyperparameter
best_hyperparameter = [0.0, 0.0]  # [nu, gamma]
f3_score_best = 0.0
param_grid = {'n_estimators': [200, 700],
              'max_features': ['auto', 'sqrt', 'log2']}
# param_grid = {'n_estimators': [200], 'max_features': ['auto']} # for test
grid = ParameterGrid(param_grid)

from sklearn.ensemble import RandomForestClassifier

for params in grid:
    f3_score_total = 0.0
    for train_index, test_index in kf.split(X):
        #print('TRAIN:', train_index, 'TEST:', test_index)
        tmp = X.iloc[train_index]
        X_train = tmp.drop(['Y_label'], 1)
        Y_train = tmp['Y_label']
        X_test = X.drop(['Y_label'], 1).iloc[test_index]
        Y_test = X['Y_label'].iloc[test_index].values
        rfc = RandomForestClassifier(n_jobs=-1, max_features=params['max_features'],
                                     n_estimators=params['n_estimators'], oob_score=True)
        rfc.fit(X_train, Y_train)
        X_pred = rfc.predict(X_test)

        # print(X_pred)
        # print(Y_test)

        from sklearn.metrics import accuracy_score
        from sklearn.metrics import classification_report
        #print(accuracy_score(Y_test, X_pred, normalize=False))
        #print(classification_report(Y_test, X_pred))

        precision, recall, f3_score, support = precision_recall_fscore_support(
            Y_test, X_pred, beta=3.0)
        f3_score_total += f3_score[1]

    print("f3_score = {}".format(f3_score_total / 5))
    if f3_score_total / 5 > f3_score_best:
        f3_score_best = f3_score_total / 5
        best_hyperparameter[0], best_hyperparameter[
            1] = params['max_features'], params['n_estimators']

print("\n\nf3_score_best = {}".format(f3_score_best))
print("best hyperparameter nu = {}".format(best_hyperparameter[0]))
print("best hyperparameter gamma = {}".format(best_hyperparameter[1]))
