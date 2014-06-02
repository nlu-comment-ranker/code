##
# Social Web Comment Ranking
# CS224U Spring 2014
# Stanford University 
#
# Feature Extraction
# methods and classes
#
# Ian F. Tenney
# May 24, 2014
##

# import collections
from collections import Counter
import itertools
import time
import sys

import nltk
import hyphen
import re

from numpy import linalg
from numpy import array, sqrt, log, nan

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

###########################
# Static Helper Functions #
###########################

##
# Readability scores
def is_polysyllabic(w, hyphenator=hyphen.Hyphenator('en_US')):
    if len(w) > 30: return False
    return (len(hyphenator.syllables(unicode(w))) >= 3)

def SMOG(words, sentences):
    if type(words) == Counter:
        npoly = sum([c for w,c in words.items() if is_polysyllabic(w)])
    else: # words as list
        npoly = len(filter(words,is_polysyllabic))
    val = npoly*30.0/len(sentences)
    grade = 1.0430 * sqrt(val) + 3.1291
    return grade

##
# Distributional models

# def counter_dot_product(c1,c2, norm=False):
#     """Compute the dot product of two sparse vectors (as dicts).
#     If norm=True, then will normalize each before computing,
#     to yield the cosine distance."""
#     c1_f = {k:v for k,v in c1.items() if k in c2}
#     c2_f = {k:v for k,v in c2.items() if k in c1}
#     dp = sum([c1_f[k]*c2_f[k] for k in c1_f.keys()])
#     if norm == True:
#         nc = linalg.norm(c1.values()) * linalg.norm(c2.values())
#     else: nc = 1.0
#     return dp/nc


# def counter_L1_norm(c):
#     """Normalize a counter by sum of elements, 
#     i.e. to convert to a probability distribution."""
#     total = sum(c.values())
#     return {k:(v*1.0/total) for k,v in c.items()}

def entropy_normalized(v):
    """Calculate the length-normalized entropy 
    of a (sparse) word distribution vector."""
    vnz = array(v[v.nonzero()]).reshape((-1,)) # ensure 1D
    Z = 1.0*sum(vnz)
    vdist = vnz / Z
    return (-1/Z)*( sum(vdist*log(vdist)) - log(Z) )

def sparse_row_cosine_distance(v1,v2):
    """Compute the cosine distance between two sparse row vectors."""
    dp = (v1 * v2.T)[0,0]
    norm = sqrt(1.0*(v1 * v1.T)*(v2 * v2.T))[0,0]
    return (dp / norm if norm != 0 else 0)

##
# Context-based features
# operate on FeatureSet.parent (i.e. a Submission object)
# with a list of comments to order
def rank_comments(sub):
    if hasattr(sub, 'comments_ranked') and sub.comments_ranked:
        return
    
    # Rank comments by timestamp: get position_rank
    sub.comments.sort(key=lambda c: c.timestamp)
    for i,c in enumerate(sub.comments): 
        c.position_rank = i

    # Rank comments by score, high -> low
    # get score rank
    sub.comments.sort(key=lambda c: c.score, reverse=True)
    for i,c in enumerate(sub.comments): 
        c.score_rank = i

    sub.comments_ranked = True    

def attach_token_stem_features(obj):
    """
    Generate a FeatureSet containing stemmed token vector,
    and attach it to the given object.
    Used e.g. to compute word vector for a submission, for 
    later reference by comments in calculating similarity.
    """
    if hasattr(obj, 'featureSet'):
        return

    f = FeatureSet(obj)
    f.tokenize()
    f.stem()
    obj.featureSet = f

def init_featureset(obj):
    if hasattr(obj, 'featureSet'):
        return
    obj.featureSet = FeatureSet(obj)


######################
# Vector Space Model #
######################

# Tokenizers for input
sent_tokenize = nltk.tokenize.sent_tokenize
word_tokenize = nltk.tokenize.word_tokenize
stemmer = nltk.stem.porter.PorterStemmer()

def wordfilter(word):
    """Crude filter to skip punctuation and URLs
    Note that this leaves in the 'http' token, which
    can still be used as a proxy for URLs."""
    if len(word) < 2: return False
    if "www" in word: return False
    # if not a.isalpha(): return False
    return True

def default_tokenizer(text, wf=wordfilter):
    sentences = sent_tokenize(text)
    words = (word_tokenize(s) for s in sentences) # lazy generator
    # To-Do: make this filtering more robust, add special tokens / transform
    words_filtered = (w for w in itertools.chain(*words) if wf(w)) # lazy generator
    return [stemmer.stem(w) for w in words_filtered]


class VSM(object):

    vectorizer = None
    tfidf_transformer = None

    ##
    # Data
    featureSets = None # FeatureSet objects
    parentFeatureSets = None # Parent FeatureSet objects

    texts = None # comment texts
    parent_texts = None # parent texts

    # Global VSM (big sparse matricies)
    wcMatrix = None
    tfidfMatrix = None

    def __init__(self, featureSets):
        self.featureSets = featureSets

        # All comment texts
        self.texts = [f.original.text for f in self.featureSets]
        
        # Collect unique parents
        parents_unique = list({f.parent for f in self.featureSets})
        for p in parents_unique: 
            init_featureset(p)
        self.parentFeatureSets = [p.featureSet for p in parents_unique]
        
        # All parent texts
        self.parent_texts = [p.original.text for p in self.parentFeatureSets]

    def index_featuresets(self, tag=""):
        """Tag all featuresets with a reference to this VSM,
        and a row index for accessing wcMatrix, tfidfMatrix, etc."""
        # Tag each featureSet with index into VSM
        allfeatures = self.featureSets + self.parentFeatureSets
        for i,f in enumerate(allfeatures):
            """Store with custom tag, to allow a featureSet to
            belong to multiple VSMs"""
            setattr(f, "VSM"+tag, self) # store reference to VSM
            setattr(f, "vsIndex"+tag, i) # store row index
            # f.VSM = self # store reference to VSM
            # f.vsIndex = i # store row index


    def build_VSM(self, tokenizer=default_tokenizer, **voptions):
        """Generate a VSM in sparse matrix format, 
        consisting of word frequencies for each text."""
        self.vectorizer = CountVectorizer(tokenizer=tokenizer,
                                          **voptions)

        # Concatenate texts to build global VSM
        t0 = time.time()
        print >> sys.stderr, "Building VSM...",
        alltexts = self.texts + self.parent_texts
        self.wcMatrix = self.vectorizer.fit_transform(alltexts)
        print >> sys.stderr, "Completed in %.02g seconds." % (time.time() - t0)


    def build_TFIDF(self, **toptions):
        # Init
        self.tfidf_transformer = TfidfTransformer(**toptions)

        # Fit
        t0 = time.time()
        print >> sys.stderr, "Computing TFIDF...",
        self.tfidfMatrix = self.tfidf_transformer.fit_transform(self.wcMatrix)
        print >> sys.stderr, "Completed in %.02g seconds." % (time.time() - t0)



####################
# FeatureSet Class #
####################

# Prefixer functions; convert vars_user_activity to
# vars_feature_user_[local/global]
local_prefixer = lambda name: 'user_local_' + name
global_prefixer = lambda name: 'user_global_' + name

# Dummy class for exception handling when parsing user activity
class ActivityParseError(Exception):
    pass

# Dummy class for exception handling if user or parent are missing
class MissingDataException(Exception):
    pass

class FeatureSet(object):
    # References
    original = None # comment or submission
    parent = None # reference to parent in comment tree
    user = None # reference to user; should have activity for local (subreddit) and global (all)

    ##
    # Training Labels
    vars_label = ['score',
                  'score_rank',
                  ]

    # To-Do: add alternative labels
    # e.g. score, corrected for post time

    ##
    # Intermediate/temporary features; 
    # delete these to save memory
    vars_temp = ['sentences',
                 'words',
                 'wordCounts',
                 'stemCounts',
                 'stemCounts_tfidf',
                 'posTags']


    #####################################
    # Text Features                     #
    #####################################
    # should all be numerical (or None)
    vars_feature_text = ['n_chars',                    
                         'n_words',
                         'n_sentences',
                         'n_paragraphs',
                         'n_uppercase',
                         'SMOG',
                         'n_verbs',
                         'n_nouns',
                         'entropy',
                         ]
    
    # To-Do: add distributional features
    # - Informativeness
    # - Cohesion

    ##
    # Context-based features (comment-submission)
    vars_feature_context = ['timedelta',
                            'position_rank',
                            'parent_term_overlap',
                            ]


    #########################################
    # User Features                         #
    #########################################
    # names exactly as in commentDB.py,
    # and as stored in DB, so should be
    # able to extract automatically from
    # UserActivity object using getattr()
    vars_user_activity = ['comment_count',
                          'comment_pos_karma',
                          'comment_neg_karma',
                          'comment_net_karma',
                          'comment_avg_pos_karma',
                          'comment_avg_neg_karma',
                          'comment_avg_net_karma',
                          'sub_count',
                          'sub_pos_karma',
                          'sub_neg_karma',
                          'sub_net_karma',
                          'sub_avg_pos_karma',
                          'sub_avg_neg_karma',
                          'sub_avg_net_karma',
                          ]
    # Separate lists for global, local user activity
    # (for now, same variables in each)
    vars_feature_user_local = map(local_prefixer, vars_user_activity)
    vars_feature_user_global = map(global_prefixer, vars_user_activity)

    # List of all features, for convenience
    vars_feature_all = (vars_feature_text 
                        + vars_feature_context
                        + vars_feature_user_local 
                        + vars_feature_user_global)

    def __init__(self, original, parent=None, user=None):
        """
        Basic constructor. Initializes references 
        to comment object (original) and parent,
        and sets other fields to None.
        """
        self.original = original
        self.parent = parent
        self.user = user

        # Initialize labels, from original
        # for now, this is just 'score'
        for name in self.vars_label:
            setattr(self, name, None)
        self.score = self.original.score

        # Initialize temp vars as None
        for name in self.vars_temp:
            setattr(self, name, None)

        # Initialize features as None
        for name in self.vars_feature_all:
            setattr(self, name, None)


    def __repr__(self):
        # Truncate comment description
        return "FeatureSet: " + self.original.__repr__()[:68]


    def clean_temp(self):
        """
        Remove temp variables, to conserve memory.
        Call this after processing all features, to
        avoid storing extra copies of the text.
        """
        for name in self.vars_temp:
            setattr(self, name, None)

    def to_list(self, names=(vars_feature_all+vars_label)):
        """Convert features to a flat list."""
        none_to_nan = lambda x: x if x != None else nan
        fs = [getattr(self, name) for name in names]
        return map(none_to_nan, fs)

    #################
    # User Features #
    #################

    # Internal helper function
    def _parse_UserActivity(self, ua):
        """
        Load features from a UserActivity object.
        Only reads features if UserActivity is either
        GLOBAL or for a subreddit matching original.
        If no match, throws an exception.
        """
        varnames = self.vars_user_activity
        if ua.subreddit_id == "GLOBAL":
            # Global (all of reddit)
            name_prefixer = global_prefixer
        elif ua.subreddit_id == self.original.subreddit_id:
            # Local (matching this comment)
            name_prefixer = local_prefixer
        else: 
            msg = "Error: subreddit id \"%s\" does not match expected (GLOBAL or \"%s\"." % (ua.subreddit_id, self.original.subreddit_id)
            raise ActivityParseError(msg)

        # Copy values by name, automagically!
        for name in varnames:
            val = getattr(ua, name)
            target = name_prefixer(name)
            # print "DEBUG: setting %s -> %s" % (target, repr(val))
            setattr(self, target, val)


    ##
    # TO-DO: Make this robust to missing user data,
    # i.e. if a user object given but no activity data
    ##
    def calc_user_activity_features(self):
        """Read in all available user activity stats."""
        if self.user == None:
            raise MissingDataException("FeatureSet.user not specified, unable to ")

        for ua in self.user.activities:
            try:
                self._parse_UserActivity(ua)
            except ActivityParseError as p:
                # For now, ignore other subreddits
                print str(p),
                print " : ignoring activity for %s on %s" % (ua.user_name, ua.subreddit_name)

    #################
    # Text features #
    #################

    ##
    # TO-DO:
    # - skip URLs and other non-words
    #   - even better, separate out + count! (-> use as feature)
    # - strip markdown punctuation (e.g. **word)

    # Use default tokenizers:
    # PunktSentenceTokenizer (pre-trained) for sentences
    # TreebankWordTokenizer for words (per-sentence)
    # from nltk.tokenize import word_tokenize, sent_tokenize
    def tokenize(self, 
                 sent_tokenize = nltk.tokenize.sent_tokenize,
                 word_tokenize = nltk.tokenize.word_tokenize,
                 ):
        """Tokenize text, to populate temp vars:
        - sentences
        - words
        - wordCounts
        """
        self.sentences = sent_tokenize(self.original.text)
        self.words = [word_tokenize(t) for t in self.sentences]
        self.wordCounts = Counter(itertools.chain(*self.words))

    def calc_token_counts(self):
        """
        Compute the following features:
        - n_chars
        - n_words
        - n_sentences
        - n_paragraphs
        - n_uppercase
        Requires that self.tokenize() has been called
        """
        basetext = self.original.text.strip()

        self.n_chars = len(basetext)
        self.n_words = sum(self.wordCounts.values())
        self.n_sentences = len(self.sentences)
        self.n_paragraphs = basetext.count('\n')
        self.n_uppercase = sum([c for w,c in self.wordCounts.iteritems() if w[0].isupper()])

    def calc_SMOG(self):
        self.SMOG = SMOG(self.wordCounts, self.sentences)


    ##
    # Part-of-Speech (POS) tagging, and associated features
    # - n_nouns
    # - n_verbs
    def pos_tag(self, tagger=nltk.tag.pos_tag):
        """Calculate part-of-speech tags."""
        self.posTags = map(tagger, self.words)

    def calc_nouns_verbs(self):
        """Count all nouns and verbs, from Penn Treebank POS tags."""
        self.n_nouns = len([w for w,t in itertools.chain(*self.posTags)
                            if t.startswith("NN")])
        self.n_verbs = len([w for w,t in itertools.chain(*self.posTags)
                            if t.startswith("VB")])


    ##
    # Distributional features
    # requires an associated VSM matching tag

    def getVSMfromTag(self, vsmTag=""):
        vsm = getattr(self, "VSM"+vsmTag) # vsm
        i = getattr(self, "vsIndex"+vsmTag) # index
        return vsm, i

    def calc_entropy(self, vsmTag=""):
        vsm, i = self.getVSMfromTag(vsmTag)

        # Calculate entropy from wcMatrix[i]
        self.entropy = entropy_normalized(vsm.wcMatrix[i])

    ####################
    # Context Features #
    ####################
    def calc_parent_rank_features(self):
        """
        Compute ranking features based on parent.
        - score_rank (label)
        - position_rank
        - timedelta

        Rankings are computed once for each parent,
        and memoized on self.parent and self.original
        """
        if self.parent == None:
            raise MissingDataException("FeatureSet.parent not specified, unable to calculate context features.")

        rank_comments(self.parent)
        self.score_rank = self.original.score_rank
        self.position_rank = self.original.position_rank
        self.timedelta = (self.original.timestamp - self.parent.timestamp).total_seconds()

    def gen_parent_tokens(self):
        """Generate and attach a FeatureSet to the parent object,
        consisting of token sets only."""
        if self.parent == None:
            raise MissingDataException("FeatureSet.parent not specified, unable to calculate context features.")
        attach_token_stem_features(self.parent)

    def calc_parent_overlap(self, vsmTag="", matrix="tfidfMatrix"):
        if self.parent == None:
            raise MissingDataException("FeatureSet.parent not specified, unable to calculate context features.")
        elif not hasattr(self.parent, 'featureSet'):
            raise MissingDataException("FeatureSet.parent.featureSet not specified, unable to calculate context features.")

        vsm, i = self.getVSMfromTag(vsmTag)
        _, parent_i = self.parent.featureSet.getVSMfromTag(vsmTag)
        vsMatrix = getattr(vsm, matrix)

        v = vsMatrix[i]
        p = vsMatrix[parent_i]
        overlap = sparse_row_cosine_distance(v,p)

        self.parent_term_overlap = overlap


    # def calc_parent_overlap(self):
    #     """Compute term vector overlap with parent."""
    #     if self.parent == None:
    #         raise MissingDataException("FeatureSet.parent not specified, unable to calculate context features.")
    #     elif not hasattr(self.parent, 'featureSet'):
    #         raise MissingDataException("FeatureSet.parent.featureSet not specified, unable to calculate context features.")

    #     # Compute cosine similarity
    #     cs = counter_dot_product(self.stemCounts, 
    #                              self.parent.featureSet.stemCounts, 
    #                              norm=True)
    #     self.parent_term_overlap = cs
