from argparse import ArgumentParser
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import math
import time
import random


# Removes rows with NaN or inf in specified columns
def clean_data(data, columns):
    for column in columns:
        data = data[data[column].notnull()]
    return data


# Given Pandas dataframe, returns 80/20 split into dev and test sets
# Usage:
#   dev_data, test_data = split_data(data)
def split_data(data, limit_data=False):
    if limit_data:
        random.seed(42)
        sids = random.sample(data.sid.unique(), 1000)
        data = data[data.sid.map(lambda x: x in sids)]
    dev_sid, eval_sid = train_test_split(data.sid.unique(), test_size=0.9, random_state=42)
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
        'epsilon': [0.1],
        'tol': [1e-1]}
    svr = SVR()
    cv_split = KFold(len(train_target), n_folds=10, random_state=42)
    grid_search = GridSearchCV(svr, parameters, cv=cv_split, n_jobs=8)
    grid_search.fit(train_data, train_target)

    params = grid_search.best_params_ 
    estimator = grid_search.best_estimator_
    print 'Number of support vectors: %d' % len(estimator.support_vectors_)
    return params, estimator


def _dcg(scores, k):
    if len(scores) > k:
        scores = scores[:k]
    dcgs = np.cumsum([s / math.log(1.0 + i, 2.0) for i, s in enumerate(scores, start=1)])
    if len(dcgs) < k:
        dcgs = np.append(dcgs, [dcgs[-1] for i in range(k - len(dcgs))])
    return dcgs


def ndcg(data, k):
    scores = np.zeros(k)
    skipped_submissions = 0
    for sid in data.sid.unique():
        # Add ranks and favorability scores to data frame (Hsu et al.)
        comments = data[data.sid == sid]
        if len(comments) == 0:
            skipped_submissions += 1
            continue

        ranks = range(1, len(comments) + 1)
        comments = comments.sort('pred', ascending=False)
        comments['pred_rank'] = ranks
        comments['pred_fav'] = len(comments) - comments[['pred_rank']] + 1
        comments = comments.sort('score', ascending=False)
        comments['rank'] = ranks
        comments['fav'] = len(comments) - comments[['rank']] + 1
        
        dcgs = _dcg(comments.pred_fav, k)
        idcgs = _dcg(comments.fav, k)
        scores += (dcgs / idcgs)

    return scores / (len(data.sid.unique()) - skipped_submissions)


if __name__ == '__main__':
    parser = ArgumentParser(description='Run SVR')
    parser.add_argument('datafile', type=str, help='HDF5 data file')
    parser.add_argument('-f', '--features', type=str, nargs='+', help='Features list')
    parser.add_argument('-l', '--list-features', action='store_true', default=False,
                        help='Show possible features', dest='list_features')
    parser.add_argument('-L', '--limit-data', action='store_true', default=False,
                        help='Limit to 1000 submissions', dest='limit_data')
    args = parser.parse_args()

    data = pd.read_hdf(args.datafile, 'data')
    print 'Original data dims: ' + str(data.shape)
    if args.list_features:
        print '\n'.join(data.columns.values)
        exit(0)
    columns = args.features
    data = clean_data(data, columns)
    print 'Cleaned data dims: ' + str(data.shape)
    train_df, test_df = split_data(data, args.limit_data)
    train_mat = train_df.filter(columns + ['cid', 'sid', 'score']).as_matrix()
    test_mat = test_df.filter(columns + ['cid', 'sid', 'score']).as_matrix()

    print "Training set: %d examples" % (train_mat.shape[0],)
    print "Test set: %d examples" % (test_mat.shape[0],)
    print "Selected %d features" % (len(columns),)
    print 'Features: ' + ' '.join(columns)

    train_data = train_mat[:, :-3].astype(np.float)
    train_target = train_mat[:, -1].astype(np.float)
    test_data = test_mat[:, :-3].astype(np.float)
    test_target = test_mat[:, -1].astype(np.float)

    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    start = time.time()
    params, svr = train_optimal_classifier(train_data, train_target)
    print params
    print "Took %.2f minutes to train" % ((time.time() - start) / 60.0)

    train_pred = svr.predict(train_data)
    data_pred = pd.DataFrame(np.column_stack((train_mat[:, -3:], train_pred)),
                             columns=['cid', 'sid', 'score', 'pred'])
    print 'Performance on training data'
    for i, score in enumerate(ndcg(data_pred, 20), start=1):
        print '\tNDCG@%d: %.5f' % (i, score) 
    print 'Karma MSE: %.5f' % mean_squared_error(train_mat[:, -1], train_pred)

    test_pred = svr.predict(test_data)
    data_pred = pd.DataFrame(np.column_stack((test_mat[:, -3:], test_pred)), 
                             columns=['cid', 'sid', 'score', 'pred'])
    print 'Performance on test data'
    for i, score in enumerate(ndcg(data_pred, 20), start=1):
        print '\tNDCG@%d: %.5f' % (i, score) 
    print 'Karma MSE: %.5f' % mean_squared_error(test_mat[:, -1], test_pred)
