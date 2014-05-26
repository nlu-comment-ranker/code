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

import collections
import itertools
import time

import nltk
import hyphen

from numpy import linalg
from numpy import array, sqrt, log, nan

###########################
# Static Helper Functions #
###########################

##
# Readability scores
def is_polysyllabic(w, hyphenator=hyphen.Hyphenator('en_US')):
    if len(w) > 30: return False
    return (len(hyphenator.syllables(unicode(w))) >= 3)

def SMOG(words, sentences):
    if type(words) == collections.Counter:
        npoly = sum([c for w,c in words.items() if is_polysyllabic(w)])
    else: # words as list
        npoly = len(filter(words,is_polysyllabic))
    val = npoly*30.0/len(sentences)
    grade = 1.0430 * sqrt(val) + 3.1291
    return grade

##
# Distributional models
def counter_dot_product(c1,c2, norm=False):
    """Compute the dot product of two sparse vectors (as dicts).
    If norm=True, then will normalize each before computing,
    to yield the cosine distance."""
    c1_f = {k:v for k,v in c1.items() if k in c2}
    c2_f = {k:v for k,v in c1.items() if k in c1}
    dp = sum([c1_f[k]*c2_f[k] for k in c1_f.keys()])
    if norm == True:
        nc = linalg.norm(c1.values()) * linalg.norm(c2.values())
    else: nc = 1.0
    return dp/nc


def counter_L1_norm(c):
    """Normalize a counter by sum of elements, 
    i.e. to convert to a probability distribution."""
    total = sum(c.values())
    return {k:(v*1.0/total) for k,v in c.items()}

def counter_entropy(c):
    """Basic entropy, treating counter as a 
    fully-specified multinomial distribution."""
    cn = array(counter_to_distribution(c).values())
    return -1*sum(cn * log(cn))

def counter_entropy_normalized(c):
    """
    Calculate the entropy of a word counter.
    Normalized by total word count, as in Hsu, et al.
    """
    wordcount = 1.0*sum(c.values())
    cn = array(c.values())/wordcount
    return (-1/wordcount)*(sum(cn * log(cn)) - log(wordcount))

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

class FeatureSet(object):
    # References
    original = None # comment or submission
    parent = None # reference to parent in comment tree
    user = None # reference to user; should have activity for local (subreddit) and global (all)

    ##
    # Training Labels
    vars_label = ['score',]

    # To-Do: add alternative labels
    # e.g. score, corrected for post time

    ##
    # Intermediate/temporary features; 
    # delete these to save memory
    vars_temp = ['sentences',
                 'words',
                 'wordCounts',
                 'stemCounts',
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

    # To-Do: add context features
    # - Comment-submission overlap
    # - Time since submission (critical!!!)

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
            val = getattr(self.original, name)
            setattr(self, name, val)

        # Initialize temp vars as None
        for name in self.vars_temp:
            setattr(self, name, None)

        # Initialize features as None
        for name in self.vars_feature_text:
            setattr(self, name, None)
        for name in self.vars_feature_user_local:
            setattr(self, name, None)
        for name in self.vars_feature_user_local:
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

    def to_list(self, names=vars_feature_all):
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
        self.wordCounts = collections.Counter(itertools.chain(*self.words))

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
    # Stemming, and distributional features
    # (requires: tokenize())
    # - entropy
    #
    def stem(self, stemmer=nltk.stem.porter.PorterStemmer()):
        self.stemCounts = {stemmer.stem(k):v for k,v in self.wordCounts.items()}

    def calc_entropy(self):
        """Calculate entropy from stemmed word distribution."""
        self.entropy = counter_entropy_normalized(self.stemCounts)