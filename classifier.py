#!/usr/bin/env python

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

import os, sys, re, json
from os.path import join as pathcat
from argparse import ArgumentParser
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
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

# Find files in this folder
def dummy(): pass
import inspect
THIS_DIR = os.path.dirname(inspect.getsourcefile(dummy))
# print >> sys.stderr, "Called from %s" % THIS_DIR # DEBUG

# Configuration options
import settings
GRIDSEARCH_PARAMS = settings.GRIDSEARCH_PARAMS
FEATURE_GROUPS = settings.FEATURE_GROUPS

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
def train_optimal_classifier(train_data, train_y,
                             classifier='svr',
                             quickmode=False):
    # parameters = {
    #     'kernel': ['rbf'],
    #     'C': [0.1, 1, 10, 100, 500],
    #     'degree': [0.5, 1, 2, 3, 4, 5],
    #     'gamma': [0.0],
    #     'epsilon': [0.1],
    #     'tol': [1e-1]}
    if classifier == 'svr':
        clf = SVR()
        parameters = GRIDSEARCH_PARAMS['svr']
    elif classifier == 'rf':
        clf = RandomForestRegressor()
        parameters = GRIDSEARCH_PARAMS['rf']
    elif classifier == 'elasticnet':
        clf = ElasticNet(max_iter=10000)
        parameters = GRIDSEARCH_PARAMS['elasticnet']
    else:
        raise ValueError("Invalid classifier '%s' specified." % classifier)

    print "Grid search with model '%s'" % classifier
    print "over parameters:"
    print json.dumps(parameters, indent=4)

    if quickmode:   n_folds = 3
    else:           n_folds = 10

    cv_split = KFold(len(train_y),
                     n_folds=n_folds,
                     random_state=42)
    grid_search = GridSearchCV(clf, parameters,
                               cv=cv_split, n_jobs=8,
                               verbose=1)
    grid_search.fit(train_data, train_y)

    params = grid_search.best_params_
    clfopt = grid_search.best_estimator_
    return params, clfopt

def get_feature_importance(clf, clfname,
                           feature_names = None,
                           sorted = True):
    if clfname == 'rf':
        fi = clf.feature_importances_
    elif clfname == 'elasticnet':
        fi = clf.coef_
    else:
        return ["  "], [0] # unable to compute, e.g. SVR

    if feature_names == None:
        fnames = range(0,len(fi))
    else:
        fnames = feature_names

    if sorted:
        idx = np.argsort(fi**2)[::-1] # descending order of magnitude
        fi = fi[idx]
        fnames = np.array(fnames)[idx]

    return fnames, fi


def standard_experiment(train_df, test_df, feature_names, args):

    train_df['set'] = "train" # annotate
    test_df['set'] = "test" # annotate

    # Split into X, y for regression
    target = args.target
    train_X = train_df.filter(feature_names).as_matrix().astype(np.float) # training data
    train_y = train_df.filter([target]).as_matrix().astype(np.float) # training labels
    test_X = test_df.filter(feature_names).as_matrix().astype(np.float) # test data
    test_y = test_df.filter([target]).as_matrix().astype(np.float) # ground truth

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
    print "== Finding optimal classifier using Grid Search =="
    params, clf = train_optimal_classifier(train_X, train_y,
                                           classifier=args.classifier,
                                           quickmode=args.quickmode)
    print "Optimal parameters: " + json.dumps(params, indent=4)
    if hasattr(clf, "support_vectors_"):
        print 'Number of support vectors: %d' % len(clf.support_vectors_)
    print "Took %.2f minutes to train" % ((time.time() - start) / 60.0)

    ##
    # Set up evaluation function
    if args.ndcg_weight == 'target':
        favfunc = evaluation.fav_target # score weighting
    else:
        favfunc = evaluation.fav_linear # rank weighting

    max_K = 20
    eval_func = lambda data: evaluation.ndcg(data, max_K,
                                             target=args.ndcg_target,
                                             result_label=result_label,
                                             fav_func=favfunc)

    ##
    # Predict scores for training set
    result_label = "pred_%s" % args.target # e.g. pred_score
    train_pred = clf.predict(train_X)
    train_df[result_label] = train_pred

    print 'Performance on training data (NDCG with %s weighting)' % args.ndcg_weight
    ndcg_train = eval_func(train_df)
    for i, score in enumerate(ndcg_train, start=1):
        print '\tNDCG@%d: %.5f' % (i, score)
    print 'Karma MSE: %.5f' % mean_squared_error(train_y, train_pred)

    ##
    # Predict scores for test set
    test_pred = clf.predict(test_X)
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
            pickle.dump(clf, f)

    ##
    # Get feature importance, if possible
    if args.savename:
        feature_importances = get_feature_importance(clf, args.classifier,
                                                     feature_names=feature_names,
                                                     sorted=True)
        saveas = args.savename + ".topfeatures.txt"
        print "== Recording top features to %s ==" % saveas
        # np.savetxt(saveas, feature_importances)
        # with open(saveas, 'w') as f:
            # json.dump(feature_importances, f, indent=2)
        with open(saveas, 'w') as f:
            maxlen = max([len(fname) for fname in feature_importances[0]])
            f.write("# Model: %s\n" % args.classifier)
            f.write("# Params: %s\n" % json.dumps(params))
            for fname, val in zip(*feature_importances):
                f.write("%s  %.06f\n" % (fname.ljust(maxlen), val))
            f.flush()

    ##
    # Save data to HDF5
    if args.savename:

        # Save score predictions
        fields = ["self_id", "parent_id", 'cid', 'sid', 'set',
                  args.target, result_label]
        if not args.ndcg_target in fields:
            fields.append(args.ndcg_target)
        saveas = args.savename + ".scores.h5"
        print "== Saving raw predictions as %s ==" % saveas
        outdf = pd.concat([train_df[fields], test_df[fields]],
                          ignore_index=True)
        outdf.to_hdf(saveas, 'data')

        if args.savefull:
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


def prep_data_standard(data, feature_names, args):

    data = clean_data(data, feature_names)
    print 'Cleaned data dims: ' + str(data.shape)

    # Split into train, test
    # and select training target
    target = args.target
    train_df, test_df = split_data(data, args.limit_data,
                                   args.test_fraction)
    return train_df, test_df


def prep_data_crossdomain(data_train, data_test, feature_names, args):
    data_train = clean_data(data_train, feature_names)
    print 'TRAIN: Cleaned data dims: ' + str(data_train.shape)

    data_test = clean_data(data_test, feature_names)
    print 'TEST: Cleaned data dims: ' + str(data_test.shape)

    train_df = data_train
    test_df = data_test
    return train_df, test_df


def main(args):

    # Load Data File
    data = pd.read_hdf(args.datafile, 'data')

    print 'Original data dims: ' + str(data.shape)
    if args.list_features:
        print '\n'.join(data.columns.values)
        exit(0)

    # Select Features and trim data so all features present
    feature_names = set()
    for fgname in args.feature_groups:
        feature_names.update(FEATURE_GROUPS[fgname])
    for fname in args.feature_names:
        feature_names.add(fname)
    feature_names = sorted(list(feature_names))
    print "Using features: \n  " + "\n  ".join(feature_names)

    if args.crossdomain != "":
        print "== Running cross-domain experiment =="
        print "  TRAIN: %s" % args.datafile
        print "   TEST: %s" % args.crossdomain
        data_x = pd.read_hdf(args.crossdomain, 'data')
        train_df, test_df = prep_data_crossdomain(data, data_x, feature_names, args)
        standard_experiment(train_df, test_df, feature_names, args)
    else:
        print "== Running standard experiment =="
        train_df, test_df = prep_data_standard(data, feature_names, args)
        standard_experiment(train_df, test_df, feature_names, args)

if __name__ == '__main__':
    parser = ArgumentParser(description='Run SVR')

    parser.add_argument('datafile', type=str,
                        help='HDF5 data file')

    parser.add_argument('--crossdomain', type=str,
                        default="",
                        help='Cross-domain test dataset')

    parser.add_argument('-s', '--savename', dest='savename',
                        default=None,
                        help="Name to save model and results. Extensions (.model.pkl and .data.h5) will be added.")

    parser.add_argument('--savefull', dest='savefull',
                        action="store_true",
                        help="Save full train/test set with predictions (WARNING: uses a lot of disk space).")

    parser.add_argument("-q", "--quickmode",
                        dest="quickmode",
                        action="store_true",
                        help="Quick mode for testing.")

    parser.add_argument('-f', '--features', type=str,
                        dest="feature_names", default=[],
                        nargs='+', help='Features list')

    parser.add_argument("--fg", "--featuregroup", type=str,
                        dest="feature_groups",
                        nargs="+", default=[],
                        help="Pre-specified feature groups, as given in settings.py")

    parser.add_argument('-t', '--target', dest='target',
                        type=str, default='score',
                        help="Training objective (e.g. score)")

    parser.add_argument('-c', '--classifier', default='svr',
                        type=str, choices=['svr', 'rf', 'elasticnet'],
                        help="Classifier (SVR or Random Forest or Elastic Net).")

    parser.add_argument('--list-features', action='store_true', default=False,
                        help='Show possible features', dest='list_features')

    parser.add_argument('-l', '-L', '--limit-data', dest='limit_data',
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

    parser.add_argument('--n_target', dest='ndcg_target',
                        default='score', type=str,
                        help="""
                        Scoring field when using 'target' mode NDCG;
                        useful if regressing on log_score or other.
                        """)

    args = parser.parse_args()


    # main(args)

    import pdb
    try:
        main(args)
    except Exception as e:
        print repr(e)
        pdb.post_mortem()