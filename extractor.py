#!/usr/bin/env python

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

# Functional take(l,n), evaluated lazily with generators
def take_n(gen, n):
    return (gen.next() for i in xrange(n))


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

##
# TO-DO:
# make this more memory-efficient;
# discard the local VSMs after computing comment features
def build_thread_VSMs(vsm_global):
    """Construct individual VSMs for each submission."""
    for pfs in vsm_global.parentFeatureSets:
        # Restrict to children in current dataset
        child_features = [c.featureSet for c in pfs.original.comments if hasattr(c, 'featureSet')]

        # Initialize VSM
        pfs.vsm_thread = features.VSM(child_features)
        # print "Submission %s: %d/%d comments loaded" % (pfs.original.sub_id,
        #                                                 len(child_features),
        #                                                 len(pfs.original.comments))
        pfs.vsm_thread.index_featuresets(tag="_thread")
        # Copy counts from global VSM, to avoid re-tokenizing
        pfs.vsm_thread.build_VSM_from_existing(vsm_global)
        pfs.vsm_thread.build_TFIDF()
        # print "  VSM: %s" % (str(pfs.vsm_thread.tfidfMatrix.shape))

        # Clean up memory (only need to keep tfidfMatrix)
        pfs.vsm_thread.vectorizer = None
        pfs.vsm_thread.tfidf_transformer = None
        pfs.vsm_thread.wcMatrix = None


def calcGeneralFeatures(f, options, vsmTag_global="_global"):
    """Calculate user-based, context-based, and VSM features."""
    # This won't run unless user data is available
    if options.f_user:
        try:
            f.calc_user_activity_features()
            print >> options.logfile, "Loaded user for %s (user %s)" % (f.self_id, f.user.name)
        except features.MissingDataException as e:
            print >> options.logfile, "Missing user data for %s" % (f.self_id)

    f.calc_parent_rank_features()

    # Distributional: require VSMs
    f.calc_entropy(vsmTag=vsmTag_global)
    f.calc_parent_overlap(vsmTag=vsmTag_global)
    f.calc_informativeness()


def calcLocalFeatures(f, options):
    """Calculate local features, involving tokenization
    and intensive text processing."""
    f.tokenize()
    f.calc_token_counts()
    f.calc_SMOG()

    # These are slow! Enable only if needed.
    if options.f_pos:
        f.pos_tag()
        f.calc_pos()

    # Call this to recover memory!
    f.clean_temp()


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


    ########################
    # Generate Featuresets #
    # (load from database) #
    ########################
    t0 = time.time()
    print ("== Loading comments for %d submissions ==" % sub_query.count())
    featureSets = []

    t1 = time.time()
    counter = 0
    printevery = max(500, min(4000,int(0.8*sub_query.count())))
    for c in comment_gen:
        fs = features.FeatureSet(c, user=c.user, parent=c.submission)
        featureSets.append(fs)

        # Progress indicator
        counter += 1
        if counter % printevery == 0:
            temp = t1
            t1 = time.time()
            print "  last %d: %.02f s (%d loaded)" % (printevery, (t1 - temp), counter)

    # featureSets = [features.FeatureSet(c, user=c.user, parent=c.submission) for c in comment_gen]
    print "  [loaded %d comments in %.02g seconds]" % (len(featureSets), time.time() - t0)

    ##############
    # Build VSMs #
    ##############
    vsmTag_global = "_global"
    t0 = time.time()
    print "== Generating global VSM =="
    vsm_global = construct_VSM(featureSets, vsmTag=vsmTag_global)
    print "  [completed in %.02g seconds]" % (time.time() - t0)

    # Build per-thread VSMs
    t0 = time.time()
    print "== Generating per-thread VSMs =="
    build_thread_VSMs(vsm_global)
    print "  [completed in %.02g seconds]" % (time.time() - t0)


    ############################
    # Process General Features #
    ############################
    t0 = time.time()
    t1 = time.time()
    counter = 0
    printevery = len(featureSets) / 10
    print "== Extracting general features: %d total comments ==" % len(featureSets)
    for f in featureSets:
        calcGeneralFeatures(f, options, vsmTag_global=vsmTag_global)
        # calcLocalFeatures(f, options)

        # Progress indicator
        counter += 1
        if counter % printevery == 0:
            temp = t1
            t1 = time.time()
            print "  -> last %d: %.02f s (%.01f%% done)" % (printevery, (t1 - temp), counter*100.0/len(featureSets))

    dt = time.time() - t0
    print "  [completed %d in %.02f s]" % (counter, dt)
    print "  (%d ms per comment)" % ((dt * 1000)/counter)

    ##########################
    # Process Local Features #
    ##########################
    t0 = time.time()
    t1 = time.time()
    counter = 0
    printevery = len(featureSets) / 10
    if options.f_pos: printevery /= 10 # much slower!
    printevery = max(100, printevery) # avoid dumping too much text :)
    print "== Extracting local features: %d total comments ==" % len(featureSets)
    for f in featureSets:
        # calcGeneralFeatures(f, options, vsmTag_global=vsmTag_global)
        calcLocalFeatures(f, options)

        # Progress indicator
        counter += 1
        if counter % printevery == 0:
            temp = t1
            t1 = time.time()
            print "  -> last %d: %.02f s (%.01f%% done)" % (printevery, (t1 - temp), counter*100.0/len(featureSets))

    dt = time.time() - t0
    print "  [completed %d in %.02f s]" % (counter, dt)
    print "  (%d ms per comment)" % ((dt * 1000)/counter)


    #########################
    # Convert and Save Data #
    #########################

    # Convert to DataFrame
    print "== Converting to DataFrame =="
    df = features.fs_to_DataFrame(featureSets)
    df = features.derive_features(df)

    # Convert all unicode to ASCII strings before saving to HDF5
    cols_unicode = ['self_id', 'parent_id', 'distinguished']
    for name in cols_unicode:
        df[name] = map(str, df[name])
    # df['self_id'] = map(str, df['self_id'])
    # df['parent_id'] = map(str, df['parent_id'])

    # Convert all boolean to float (0.0, 1.0), NaN if missing
    cols_boolean = ['is_mod', 'is_gold', 'has_verified_email']
    for name in cols_boolean:
        df[name] = map(float, df[name])

    # For now, everything in featureSets is a comment,
    # and all parents are submissions
    # rename columns to keep Sammy happy
    df['cid'] = df['self_id']
    df['sid'] = df['parent_id']

    # Save data to HDF5
    print "== Exporting to HDF5 =="
    df.to_hdf(options.savename, "data")
    print "  [saved as %s]" % options.savename


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

    # parser.add_argument('--saveas', dest='saveas',
    #                     default='hdf',
    #                     help="Output file format.")

    ##
    # Feature Options
    # parser.add_argument('--f_user', dest='f_user',
    #                     action='store_true',
    #                     help="Include user features. Requires UserActivity to be present, and will filter database accordingly.")
    parser.add_argument('--skip_user', dest='f_user',
                        action='store_false',
                        help="Ignore user features.")

    parser.add_argument('--f_pos', dest='f_pos',
                        action='store_true',
                        help="Include POS-based features. Significantly increases computation time.")

    parser.add_argument('--N_subs', dest='N_subs',
                        type=int, default=-1,
                        help="Number of submissions to include. If -1 (default), includes all subs.")

    parser.add_argument('--logfile',
                        dest='logfile',
                        default=sys.stderr,
                        help="Filename for logging output")

    options = parser.parse_args()

    if type(options.logfile) == str:
        options.logfile = open(options.logfile, 'w')

    # main(options)

    import pdb
    try:
        main(options)
    except Exception as e:
        print repr(e)
        pdb.post_mortem()