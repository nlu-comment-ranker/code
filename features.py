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
from scipy import sparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Text Preprocessing Functions
import text_pre

###########################
# Static Helper Functions #
###########################

# Like itertools.chain, but doesn't 
# require eager argument expansion
# at the top level
def lazy_chain(listgen):
    for l in listgen:
        for e in l:
            yield e

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

def entropy_normalized(v):
    """Calculate the length-normalized entropy 
    of a (sparse) word distribution vector."""
    if v.nnz < 1: return 0 # in case of empty comment

    vnz = array(v[v.nonzero()]).reshape((-1,)) # ensure 1D
    Z = 1.0*sum(vnz)
    vdist = vnz / Z
    return (-1/Z)*( sum(vdist*log(vdist)) - log(Z) )

def sparse_row_cosine_similarity(v1,v2):
    """Compute the cosine similarity between two sparse row vectors."""
    dp = (v1 * v2.T)[0,0]
    norm = sqrt(1.0*(v1 * v1.T)*(v2 * v2.T))[0,0]
    return (dp / norm if norm != 0 else 0)

def sparse_row_jaccard_similarity(v1,v2):
    """Compute Jaccard similarity of two sparse row vectors, treating
    the nonzero indices as set members and ignoring the actual values."""
    if v1.nnz < 1 or v2.nnz < 1: return 0

    i1 = set(v1.nonzero()[1]) # nonzero column indices
    i2 = set(v2.nonzero()[1]) # nonzero column indices
    return len(i1 & i2) / (1.0*len(i1 | i2)) # Jaccard index

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

def init_featureset(obj):
    if hasattr(obj, 'featureSet'):
        return
    obj.featureSet = FeatureSet(obj)

def get_text(obj, cat_title=True):
    """Get the text from a commentDB object,
    concatenating it with the title if present."""
    if cat_title and hasattr(obj, "title"):
        return obj.title + " \n " + obj.text
    else: return obj.text

######################
# Vector Space Model #
######################

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

    def __init__(self, featureSets, tag="_global"):
        self.featureSets = featureSets

        # All comment texts
        self.texts = [get_text(f.original, cat_title=True) for f in self.featureSets]
        
        # Collect unique parents
        parents_unique = list({f.parent for f in self.featureSets})
        for p in parents_unique: 
            init_featureset(p)
        self.parentFeatureSets = [p.featureSet for p in parents_unique]
        
        # All parent texts
        self.parent_texts = [get_text(p.original, cat_title=True) for p in self.parentFeatureSets]

    def index_featuresets(self, tag="_global"):
        """Tag all featuresets with a reference to this VSM,
        and a row index for accessing wcMatrix, tfidfMatrix, etc."""
        # Tag each featureSet with index into VSM
        allfeatures = self.featureSets + self.parentFeatureSets
        self.tag = tag
        for i,f in enumerate(allfeatures):
            """Store with custom tag, to allow a featureSet to
            belong to multiple VSMs"""
            setattr(f, "VSM"+tag, self) # store reference to VSM
            setattr(f, "vsIndex"+tag, i) # store row index

    def get_row_index(self, featureSet):
        return getattr(featureSet, "vsIndex"+self.tag)


    def build_VSM_from_existing(self, vsm):
        """Generate a new VSM from an existing VSM's wcMatrix.
        Requires that all contained FeatureSets have been tagged
        with both self.tag and vsm.tag."""
        allfeatures = self.featureSets + self.parentFeatureSets
        rows = [vsm.wcMatrix[vsm.get_row_index(f)] for f in allfeatures]
        
        # Build a new wcMatrix from existing rows
        # TO-DO: convert all matricies to CSR
        self.wcMatrix = sparse.vstack(rows, format='csr')


    def build_VSM(self, tokenizer=text_pre.default_tokenizer, **voptions):
        """Generate a VSM in sparse matrix format, 
        consisting of word frequencies for each text."""
        self.vectorizer = CountVectorizer(tokenizer=tokenizer,
                                          **voptions)

        # Concatenate texts to build global VSM
        alltexts = self.texts + self.parent_texts
        self.wcMatrix = self.vectorizer.fit_transform(alltexts)

        if not sparse.isspmatrix_csr(self.wcMatrix):
            self.wcMatrix = self.wcMatrix.tocsr() # for efficient row access


    def build_TFIDF(self, **toptions):
        self.tfidf_transformer = TfidfTransformer(**toptions)
        self.tfidfMatrix = self.tfidf_transformer.fit_transform(self.wcMatrix)

        if not sparse.isspmatrix_csr(self.tfidfMatrix):
            self.tfidfMatrix = self.tfidfMatrix.tocsr() # for efficient row access



####################
# FeatureSet Class #
####################

##
# DataFrame conversion function
# operates on a list of FeatureSet objects
def fs_to_DataFrame(featureSets):
    import pandas as pd
    """Convert a list of FeatureSet objects to
    a pandas DataFrame, for convenient analysis
    and passing to ML routines."""
    colnames = (FeatureSet.vars_feature_all 
            + FeatureSet.vars_label)
    # Convert to DataFrame directly to avoid NumPy's homogeneous type requirement
    df = pd.DataFrame([f.to_list(colnames) for f in featureSets], 
                      columns=colnames)
    return df

def derive_features(df):
    """Compute derivative features from the DataFrame representation."""
    
    # Normalize score by the parent submission score
    df['score_normalized'] = df['score'] / abs(df['parent_score'])
    
    # Normalize score rank to a 0-1 scale
    # by the number of comments in a thread
    # 0 = highest, 1 = lowest
    # (note +1 to correct for zero-indexing)
    df['score_rank_normalized'] = (df['score_rank']+1) / df['parent_nchildren']

    # "Excess Relevance"

    return df

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
    # Training Labels and Metadata
    vars_label = ['score',
                  'score_rank',
                  'parent_score',
                  'parent_nchildren',
                  'self_id',
                  'parent_id',
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
                         'entropy',
                         'pos_n_noun',
                         'pos_n_nounproper',
                         'pos_n_verb',
                         'pos_n_adj',
                         'pos_n_adv',
                         'pos_n_inter',
                         'pos_n_wh',
                         'pos_n_particle',
                         'pos_n_numeral',
                         ]

    ##
    # Context-based features 
    vars_feature_context = ['timedelta',
                            'position_rank',
                            'parent_term_overlap',
                            'parent_jaccard_overlap',
                            'parent_tfidf_overlap',
                            'informativeness_thread',
                            'informativeness_global',
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

        # Backreference
        self.original.featureSet = self

        # Initialize labels, from original
        # for now, this is just 'score'
        for name in self.vars_label:
            setattr(self, name, None)
        self.score = self.original.score # raw score
        
        if hasattr(self.original, "com_id"):
            self.self_id = self.original.com_id # comment ID
        elif hasattr(self.original, "sub_id"):
            self.self_id = self.original.sub_id # submission ID

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
                 sent_tokenize = text_pre.sent_tokenize,
                 word_tokenize = text_pre.word_tokenize,
                 ):
        """Tokenize text, to populate temp vars:
        - sentences
        - words
        - wordCounts
        """
        text = get_text(self.original, cat_title=True)
        self.sentences = sent_tokenize(text)
        self.words = [word_tokenize(t) for t in self.sentences]
        # self.wordCounts = Counter(itertools.chain(*self.words))
        self.wordCounts = Counter(lazy_chain(self.words))

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
        basetext = get_text(self.original, cat_title=True).strip()

        self.n_chars = len(basetext)
        self.n_words = sum(self.wordCounts.values())
        self.n_sentences = len(self.sentences)
        self.n_paragraphs = basetext.count('\n')
        self.n_uppercase = sum([c for w,c in self.wordCounts.iteritems() if w[0].isupper()])

    def calc_SMOG(self):
        self.SMOG = SMOG(self.wordCounts, self.sentences)


    ##
    # Part-of-Speech (POS) tagging, and associated features
    # - n_noun
    # - n_verb
    def pos_tag(self, tagger=nltk.tag.pos_tag):
        """Calculate part-of-speech tags."""
        self.posTags = map(tagger, self.words)

    def calc_pos(self):
        """Count all nouns and verbs, from Penn Treebank POS tags.
        See nltl.help.upenn_tagset() for more information."""
        # alltags = [t for w,t in itertools.chain(*self.posTags)]
        alltags = [t for w,t in lazy_chain(self.posTags)]
        def count_tags(regex):
            return len([t for t in alltags if re.search(regex,t)])

        # self.n_verb = len([w for w,t in itertools.chain(*self.posTags)
        #                     if t.startswith("VB")])
        self.pos_n_noun = count_tags("NN")
        self.pos_n_nounproper = count_tags("NNP")
        self.pos_n_verb = count_tags("VB")
        self.pos_n_adj = count_tags("JJ")
        self.pos_n_adv = count_tags("RB")
        self.pos_n_inter = count_tags("UH") # interjections
        self.pos_n_wh = count_tags("WDT") + count_tags("WRB") # wh- determiners and adverbs
        self.pos_n_particle = count_tags("RP") # prepositions
        self.pos_n_numeral = count_tags("CD")


    ##
    # Distributional features
    # requires an associated VSM matching tag

    def getVSMfromTag(self, vsmTag="_global"):
        vsm = getattr(self, "VSM"+vsmTag) # vsm
        i = getattr(self, "vsIndex"+vsmTag) # index
        return vsm, i

    def calc_entropy(self, vsmTag="_global"):
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

        Also store:
        - sub_id

        Rankings are computed once for each parent,
        and memoized on self.parent and self.original
        """
        if self.parent == None:
            raise MissingDataException("FeatureSet.parent not specified, unable to calculate context features.")

        # Set parent ID
        if hasattr(self.parent, "sub_id"):
            self.parent_id = self.parent.sub_id # submission (parent) ID
        elif hasattr(self.parent, "com_id"):
            self.parent_id = self.parent.com_id # comment (parent) ID

        rank_comments(self.parent)
        self.parent_score = self.parent.score
        self.parent_nchildren = len(self.parent.comments) # total number of comments, for normalizing rankings
        self.score_rank = self.original.score_rank
        self.position_rank = self.original.position_rank
        self.timedelta = (self.original.timestamp - self.parent.timestamp).total_seconds()


    def calc_parent_overlap(self, vsmTag="_global"):
        """Calculate parent overlap in the global VSM."""
        if self.parent == None:
            raise MissingDataException("FeatureSet.parent not specified, unable to calculate context features.")
        elif not hasattr(self.parent, 'featureSet'):
            raise MissingDataException("FeatureSet.parent.featureSet not specified, unable to calculate context features.")

        vsm, i = self.getVSMfromTag(vsmTag)
        _, parent_i = self.parent.featureSet.getVSMfromTag(vsmTag)

        # Raw term-count overlap
        v = vsm.wcMatrix[i]
        p = vsm.wcMatrix[parent_i]
        overlap = sparse_row_cosine_similarity(v,p)
        self.parent_term_overlap = overlap

        # Jaccard similarity
        self.parent_jaccard_overlap = sparse_row_jaccard_similarity(v,p)

        # TF-IDF Overlap
        v = vsm.tfidfMatrix[i]
        p = vsm.tfidfMatrix[parent_i]
        overlap = sparse_row_cosine_similarity(v,p)
        self.parent_tfidf_overlap = overlap


    def calc_informativeness(self, vsmTag_thread="_thread",
                             vsmTag_global="_global"):
        """
        Calculate informativeness, i.e. the sum of the
        TF-IDF document vector in the context of the
        local thread and of the global VSM (entire dataset).
        """
        vsm, i = self.getVSMfromTag(vsmTag_thread)
        self.informativeness_thread = vsm.tfidfMatrix[i].sum()

        vsm, i = self.getVSMfromTag(vsmTag_global)
        self.informativeness_global = vsm.tfidfMatrix[i].sum()