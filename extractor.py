##
# Social Web Comment Ranking
# CS224U Spring 2014
# Stanford University 
#
# Feature Extraction
# extraction engine
#
# Ian F. Tenney
# June 1, 2014
##

import sys, time
import argparse
import pandas as pd

# Database interface
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relation, sessionmaker

# Database schema
import commentDB

# Feature library
import features

# Like itertools.chain, but doesn't 
# require eager argument expansion
# at the top level
def lazy_chain(listgen):
    for l in listgen:
        for e in l:
            yield e


def init_DB(dbfile):
    engine = create_engine("sqlite:///%s" % dbfile)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


def construct_VSM(featureSets, vsmTag="_global"):
    vsm = features.VSM(featureSets)
    vsm.index_featuresets(tag=vsmTag)
    vsm.build_VSM(lowercase=True, max_df=1.0)
    vsm.build_TFIDF()
    return vsm


def processFeatureSet(f, options, vsmTag_global="_global"):
    f.tokenize()
    f.calc_token_counts()
    f.calc_SMOG()
    
    # These are slow! Enable only if needed.
    if options.f_pos:
        f.pos_tag()
        f.calc_nouns_verbs()
    
    f.calc_entropy(vsmTag=vsmTag_global)
    
    # This won't run unless user data is available
    if options.f_user:
        f.calc_user_activity_features()

    f.calc_parent_rank_features()
    
    f.calc_parent_overlap(vsmTag=vsmTag_global)
    
    # Call this to recover memory!
    f.clean_temp()


def featureSets_to_DataFrame(featureSets):
    colnames = (features.FeatureSet.vars_feature_all 
            + features.FeatureSet.vars_label)
    # Convert to DataFrame directly to avoid NumPy's homogeneous type requirement
    df = pd.DataFrame([f.to_list(colnames) for f in featureSets], columns=colnames)

    # Convert all unicode to ASCII strings before saving to HDF5
    df['self_id'] = map(str, df['self_id'])
    df['parent_id'] = map(str, df['parent_id'])

    # For now, everything in featureSets is a comment,
    # and all parents are submissions
    # rename columns to keep Sammy happy
    df['cid'] = df['self_id']
    df['sid'] = df['parent_id']

    return df

def main(options):
    # Initialize database
    session = init_DB(options.dbfile)

    # Query submissions
    sub_query = session.query(commentDB.Submission)

    # Limit submissions
    if options.N_subs > 0:
        sub_query = sub_query.limit(options.N_subs)

    # Retrieve comments, lazily
    comment_gen = lazy_chain( (s.comments for s in sub_query) )

    # Generate featuresets
    t0 = time.time()
    print ("Loading comments for %d submissions..." % sub_query.count()),
    featureSets = [features.FeatureSet(c, user=c.user, parent=c.submission) for c in comment_gen]
    print " loaded %d in %.02g seconds." % (len(featureSets), time.time() - t0)

    # Build VSM
    vsmTag_global = "_global"
    t0 = time.time()
    print "== Generating global VSM =="
    vsm_global = construct_VSM(featureSets, vsmTag=vsmTag_global)
    print "== Completed in %.02g seconds ==" % (time.time() - t0)

    # Process Features
    t0 = time.time()
    t1 = time.time()
    counter = 0
    printevery = len(featureSets) / 10
    print "== Processing %d total comments ==" % len(featureSets)
    for f in featureSets:
        processFeatureSet(f, options, vsmTag_global=vsmTag_global)

        # Progress indicator
        counter += 1
        if counter % printevery == 0:
            temp = t1
            t1 = time.time()
            print "  last %d: %.02f s (%.01f%% done)" % (printevery, (t1 - temp), counter*100.0/len(featureSets))

    print "== Completed %d in %.02f s ==" % (counter, time.time() - t0)


    # Convert to DataFrame
    data = featureSets_to_DataFrame(featureSets)

    # Save data to HDF5
    if options.saveas == 'hdf':
        data.to_hdf(options.savename, "data")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Extractor for Reddit Comment Ranking')

    ##
    # IO Options
    parser.add_argument('--dbfile', dest='dbfile', 
                        default='redditDB.sqlite',
                        help="SQLite database file")

    parser.add_argument('--savename', dest='savename', 
                        default='data',
                        help="Output file name. Extension (e.g. .h5) will be added automatically.")

    parser.add_argument('--saveas', dest='saveas', 
                        default='hdf',
                        help="Output file format.")

    ##
    # Feature Options
    parser.add_argument('--f_user', dest='f_user',
                        action='store_true',
                        help="Include user features. Requires UserActivity to be present, and will filter database accordingly.")

    parser.add_argument('--f_pos', dest='f_pos',
                        action='store_true',
                        help="Include POS-based features. Significantly increases computation time.")

    parser.add_argument('--N_subs', dest='N_subs', 
                        type=int, default=-1,
                        help="Number of submissions to include. If -1 (default), includes all subs.")

    options = parser.parse_args()

    # main(options)

    import pdb
    try:
        main(options)
    except Exception as e:
        print repr(e)
        pdb.post_mortem()