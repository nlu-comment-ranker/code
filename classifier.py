from argparse import ArgumentParser
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np
import math
import time


# Removes rows with NaN or inf in specified columns
def clean_data(data, columns):
    raise NotImplementedError


# Given Pandas dataframe, returns 80/20 split into dev and test sets
# Usage:
#   dev_data, test_data = split_data(data)
def split_data(data):
    dev_sid, eval_sid = train_test_split(data.sid.unique(), test_size=0.2, random_state=42)
    dev_df = data[data.sid.map(lambda x: x in dev_sid)]
    eval_df = data[data.sid.map(lambda x: x in eval_sid)]
    return dev_df, eval_df


# Runs grid search to determine the best parameters to use for SVR
def train_optimal_classifier(train_data, train_target):
    parameters = {
        'kernel': ['rbf'], 
        'C': [0.1, 1, 10, 100, 500],
        'degree': [0.5, 1, 2, 3, 4, 5],
        'gamma': [0.0],
        'epsilon': [0.1]}
    svr = SVR()
    cv_split = KFold(len(train_target), n_folds=10, random_state=42)
    grid_search = GridSearchCV(svr, parameters, cv=cv_split, n_jobs=8)
    grid_search.fit(train_data, train_target)

    params = grid_search.best_params_ 
    estimator = grid_search.best_estimator_
    return params, estimator


def _dcg(scores, k):
    if len(scores) > k:
        scores = scores[:k]
    return sum(s / math.log(1.0 + i, 2.0) for i, s in enumerate(scores, start=1))


def ndcg(data, k):
    scores = []
    for sid in data.sid.unique():
        # Add ranks and favorability scores to data frame (Hsu et al.)
        comments = data[data.sid == sid]
        ranks = range(1, len(comments) + 1)
        comments = comments.sort('pred', ascending=False)
        comments['pred_rank'] = ranks
        comments['pred_fav'] = len(comments) - comments[['pred_rank']] + 1
        comments = comments.sort('score', ascending=False)
        comments['rank'] = ranks
        comments['fav'] = len(comments) - comments[['rank']] + 1
        
        dcg = _dcg([f for f in comments.pred_fav], k)
        idcg = _dcg([f for f in comments.fav], k)
        scores.append(dcg / idcg)

    return sum(scores) / float(len(scores))


if __name__ == '__main__':
    parser = ArgumentParser(description='Run SVR')
    parser.add_argument('datafile', type=str, help='HDF5 data file')
    parser.add_argument('-f', '--features', type=str, nargs='+', help='Features list')
    parser.add_argument('-l', '--list-features', action='store_true', default=False,
                        help='Show possible features', dest='list_features')
    args = parser.parse_args()

    data = pd.read_hdf(args.datafile, 'data')
    # TODO specify columns in command line
    # columns = ['n_chars', 'n_words', 'n_uppercase', 'SMOG', 'entropy', 'timedelta']
    if args.list_features:
        print data.columns.values
        exit(0)
    columns = args.features
    train_df, test_df = split_data(data)
    train_mat = train_df.filter(columns + ['cid', 'sid', 'score']).as_matrix()
    test_mat = test_df.filter(columns + ['cid', 'sid', 'score']).as_matrix()

    print "Training set: %d examples" % (train_mat.shape[0],)
    print "Test set: %d examples" % (test_mat.shape[0],)
    print "Selected %d features" % (len(columns),)
    print columns

    train_data = train_mat[:, :-3].astype(np.float)
    train_target = train_mat[:, -1].astype(np.float)
    test_data = test_mat[:, :-3].astype(np.float)
    test_target = test_mat[:, -1].astype(np.float)

    start = time.time()
    params, svr = train_optimal_classifier(train_data, train_target)
    print params
    print "Took %.2f minutes to train" % ((time.time() - start) / 60.0)

    test_pred = svr.predict(test_data)
    data_pred = pd.DataFrame(np.column_stack((test_mat[:, -3:], test_pred)), 
                             columns=['cid', 'sid', 'score', 'pred'])
    for k in [1, 5, 10, 20]:
        print 'NDCG@%d: %.5f' % (k, ndcg(data_pred, k)) 
