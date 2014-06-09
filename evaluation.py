##
# Social Web Comment Ranking
# CS224U Spring 2014
# Stanford University 
#
# Evaluation Function
#
# Sammy Nguyen
# Ian Tenney
# June 8, 2014
##

import numpy as np
import math

######################
# Evaluation Metrics #
######################

def _dcg(scores, k):
    if len(scores) > k:
        scores = scores[:k]
    dcgs = np.cumsum([s / math.log(1.0 + i, 2.0) for i, s in enumerate(scores, start=1)])
    if len(dcgs) < k:
        dcgs = np.append(dcgs, [dcgs[-1] for i in range(k - len(dcgs))])
    return dcgs

def fav_linear(comments, target, result_label):
    """Calculate favorability for NDCG on a linear scale,
    as n_comments - rank"""
    ranks = range(1, len(comments) + 1)

    comments = comments.sort(result_label, ascending=False)
    comments['pred_rank'] = ranks

    # comments['pred_fav'] = len(comments) - comments[['pred_rank']] + 1
    comments['pred_fav'] = len(comments) - comments['pred_rank'] + 1

    comments = comments.sort(target, ascending=False)
    comments['rank'] = ranks
    # comments['fav'] = len(comments) - comments[['rank']] + 1
    comments['fav'] = len(comments) - comments['rank'] + 1

    return comments

def fav_target(comments, target, result_label):
    """
    Calculate favorability for NDCG as the raw score of a comment.
    """
    real_fav = comments[target].as_matrix()

    comments = comments.sort(result_label, ascending=False)
    comments['pred_fav'] = real_fav

    comments = comments.sort(target, ascending=False)
    comments['fav'] = comments[target]
    return comments

def thread_ndcg(comments, k, target, result_label,
                fav_func=fav_linear):
    ##
    # Add ranks and favorability scores to data frame (Hsu et al.)
    comments = fav_func(comments, target, result_label)

    ##
    # Compute NDCG@i for i = 1,2,...k
    # for this submission, add to scores
    dcgs = _dcg(comments['pred_fav'], k)
    idcgs = _dcg(comments['fav'], k)
    res = (dcgs / idcgs)
    res[idcgs == 0] = 0 # ignore NaN
    return res


def ndcg(data, k, target, result_label,
         fav_func=fav_linear,
         min_comments = 2):
    scores = np.zeros(k)
    skipped_submissions = 0

    # import pdb
    # pdb.set_trace()

    # Loop through all submissions
    for sid in data.sid.unique():

        # Select comments for each submission
        comments = data[data.sid == sid]
        if len(comments) < min_comments: # skip trivial rankings
            skipped_submissions += 1
            continue

        res = thread_ndcg(comments, k, target, result_label,
                          fav_func)

        scores += res
        if not all(np.isfinite(scores)):
            import pdb
            pdb.set_trace()


    # Return average score
    return scores / (len(data.sid.unique()) - skipped_submissions)



########################################
# Evaluation Post-Processing Functions #
########################################

import pandas as pd


def gen_k_labels(max_K):
    return ["k%d" % k for k in range(1,max_K+1)]

def evaluate_submissions(data, max_K=20, target='score',
                         fav_func=fav_target):
    """
    Evaluate per-thread submissions for a given dataset.
    Returns a DataFrame with columns k##, sid, and ncomments
    """

    sids = data.sid.unique()

    ndcgs = []
    ncomments = []
    for sid in sids:
        comments = data[data.sid == sid]

        # Evaluate 
        res = thread_ndcg(comments, max_K, 
                          target, "pred_" + target,
                          fav_func=fav_func)
        ndcgs.append(res)
        ncomments.append(len(comments))

    # evaldf = pd.DataFrame(ndcgs, columns=range(1,max_K+1)).add_prefix("k")
    evaldf = pd.DataFrame(ndcgs, columns=gen_k_labels(max_K))
    evaldf['sid'] = sids
    evaldf['ncomments'] = ncomments

    return evaldf

def calc_y_yerr(evaldf, max_K):
    """
    Calculate mean and stdev_mean at each k,
    for a given dataframe returned by evaluate_submissions, above.
    """ 
    cols = gen_k_labels(max_K)
    y = evaldf[cols].mean()
    yerr = evaldf[cols].std() / np.sqrt(evaldf.shape[0] - 1)
    return y, yerr