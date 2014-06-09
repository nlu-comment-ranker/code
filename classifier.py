##
# Social Web Comment Ranking
# CS224U Spring 2014
# Stanford University 
#
# Classifier Engine
#
# Sammy Nguyen
# Ian Tenney
# June 2, 2014
##

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

# Evaluation Functions
import evaluation


# Removes rows with NaN or inf in specified columns
def clean_data(data, columns):
    for column in columns:
        data = data[data[column].notnull()]
    return data


# Given Pandas dataframe, returns 80/20 split into dev and test sets
# Usage:
#   dev_data, test_X = split_data(data)
def split_data(data, limit_data=0, test_fraction=0.9):
    if limit_data > 0:
        random.seed(42)
        sids = random.sample(data.sid.unique(), limit_data)
        data = data[data.sid.map(lambda x: x in sids)]

    # Split along unique submission IDs
    dev_sid, eval_sid = train_test_split(data.sid.unique(), 
                                         test_size=test_fraction, 
                                         random_state=42)
    dev_df = data[data.sid.map(lambda x: x in dev_sid)]
    eval_df = data[data.sid.map(lambda x: x in eval_sid)]
    return dev_df, eval_df


# Runs grid search to determine the best parameters to use for SVR
def train_optimal_classifier(train_data, train_y):
    parameters = {
        'kernel': ['rbf'], 
        'C': [0.1, 1, 10, 100, 500],
        'degree': [0.5, 1, 2, 3, 4, 5],
        'gamma': [0.0],
        'epsilon': [0.1],
        'tol': [1e-1]}
    svr = SVR()
    cv_split = KFold(len(train_y), n_folds=10, random_state=42)
    grid_search = GridSearchCV(svr, parameters, cv=cv_split, n_jobs=8)
    grid_search.fit(train_data, train_y)

    params = grid_search.best_params_ 
    estimator = grid_search.best_estimator_
    print 'Number of support vectors: %d' % len(estimator.support_vectors_)
    return params, estimator


def main(args):
    # Load Data File
    data = pd.read_hdf(args.datafile, 'data')
    
    print 'Original data dims: ' + str(data.shape)
    if args.list_features:
        print '\n'.join(data.columns.values)
        exit(0)
    
    # Select Features and trim data so all features present
    feature_names = args.features
    data = clean_data(data, feature_names)
    print 'Cleaned data dims: ' + str(data.shape)

    # Split into train, test
    # and select training target
    target = args.target
    train_df, test_df = split_data(data, args.limit_data,
                                   args.test_fraction)
    train_df['set'] = "train" # annotate
    test_df['set'] = "test" # annotate

    # Split into X, y for regression
    train_X = train_df.filter(feature_names).as_matrix().astype(np.float) # training data
    train_y = train_df.filter([target]).as_matrix().astype(np.float) # training labels
    test_X = test_df.filter(feature_names).as_matrix().astype(np.float) # test data
    test_y = test_df.filter([target]).as_matrix().astype(np.float) # ground truth

    # import pdb
    # pdb.set_trace()

    # For compatibility, make 1D
    train_y = train_y.reshape((-1,))
    test_y = test_y.reshape((-1,))

    print "Training set: %d examples" % (train_X.shape[0],)
    print "Test set: %d examples" % (test_X.shape[0],)
    print "Selected %d features" % (len(feature_names),)
    print 'Features: %s' % (' '.join(feature_names))

    ##
    # Preprocessing: scale data, keep SVM happy
    scaler = preprocessing.StandardScaler()
    train_X = scaler.fit_transform(train_X) # faster than fit, transform separately
    test_X = scaler.transform(test_X)

    ##
    # Run Grid Search / 10xv on training/dev set
    start = time.time()
    params, svr = train_optimal_classifier(train_X, train_y)
    print params
    print "Took %.2f minutes to train" % ((time.time() - start) / 60.0)

    ##
    # Set up evaluation function
    if args.ndcg_weight == 'target':
        favfunc = evaluation.fav_target # score weighting
    else:
        favfunc = evaluation.fav_linear # rank weighting

    max_K = 20
    eval_func = lambda data: evaluation.ndcg(data, max_K,
                                             target=target, 
                                             result_label=result_label,
                                             compute_favorability=favfunc)

    ##
    # Predict scores for training set
    result_label = "pred_%s" % target # e.g. pred_score
    train_pred = svr.predict(train_X)
    train_df[result_label] = train_pred

    print 'Performance on training data (NDCG with %s weighting)' % args.ndcg_weight
    ndcg_train = eval_func(train_df)
    for i, score in enumerate(ndcg_train, start=1):
        print '\tNDCG@%d: %.5f' % (i, score) 
    print 'Karma MSE: %.5f' % mean_squared_error(train_y, train_pred)

    ##
    # Predict scores for test set
    test_pred = svr.predict(test_X)
    test_df[result_label] = test_pred

    print 'Performance on test data (NDCG with %s weighting)' % args.ndcg_weight
    ndcg_test = eval_func(test_df)
    for i, score in enumerate(ndcg_test, start=1):
        print '\tNDCG@%d: %.5f' % (i, score) 
    print 'Karma MSE: %.5f' % mean_squared_error(test_y, test_pred)

    ##
    # Save model to disk
    if args.savename:
        import cPickle as pickle
        saveas = args.savename + ".model.pkl"
        print "== Saving model as %s ==" % saveas
        with open(saveas, 'w') as f:
            pickle.dump(svr, f)

    ##
    # Save data to HDF5
    if args.savename:
        # Concatenate train, test
        df = pd.concat([train_df, test_df], 
                       ignore_index=True)

        print "== Exporting data to HDF5 =="
        saveas = args.savename + ".data.h5"
        df.to_hdf(saveas, "data")
        print "  [saved as %s]" % saveas

        # Save NDCG calculations
        dd = {'k':range(1,max_K+1), 'method':[args.ndcg_weight]*max_K,
              'ndcg_train':ndcg_train, 'ndcg_test':ndcg_test}
        resdf = pd.DataFrame(dd)
        saveas = args.savename + ".results.csv"
        print "== Saving results to %s ==" % saveas
        resdf.to_csv(saveas)


if __name__ == '__main__':
    parser = ArgumentParser(description='Run SVR')

    parser.add_argument('datafile', type=str, help='HDF5 data file')

    parser.add_argument('-s', '--savename', dest='savename', 
                        default=None,
                        help="Name to save model and results. Extensions (.model.pkl and .data.h5) will be added.")

    parser.add_argument('-f', '--features', type=str, 
                        nargs='+', help='Features list')

    parser.add_argument('-t', '--target', dest='target',
                        type=str, default='score',
                        help="Training objective (e.g. score)")

    parser.add_argument('-l', '--list-features', action='store_true', default=False,
                        help='Show possible features', dest='list_features')

    parser.add_argument('-L', '--limit-data', dest='limit_data', 
                        default=0, type=int,
                        help="Limit to # of submissions")

    parser.add_argument('--tf', dest='test_fraction', 
                        default=0.9, type=float,
                        help="Fraction to reserve for testing")

    parser.add_argument('--n_weight', dest='ndcg_weight', 
                        default='target', type=str,
                        help="""
                        Weighting option for NDCG calculation.
                        'target' : weight by regression target
                        'linear' : weight by rank
                        """)

    args = parser.parse_args()


    # main(args)

    import pdb
    try:
        main(args)
    except Exception as e:
        print repr(e)
        pdb.post_mortem()