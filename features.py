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

from numpy import sqrt

###########################
# Static Helper Functions #
###########################

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


####################
# FeatureSet Class #
####################

class FeatureSet(object):
    # References
    original = None # comment or submission
    parent = None # reference to parent in comment tree

    ##
    # Intermediate/temporary features; 
    # delete these to save memory
    vars_temp = ['sentences',
                 'words',
                 'wordCounts',
                 'stemCounts',
                 'posTags']

    ##
    # Core Features
    # should all be numerical (or None)
    vars_feature = ['n_chars',                    
                    'n_words',
                    'n_sentences',
                    'n_paragraphs',
                    'n_uppercase',
                    'SMOG',
                    'n_verbs',
                    'n_nouns',
                    ]
    
    # To-Do: add distributional features
    # - Entropy
    # - Informativeness
    # - Cohesion

    # To-Do: add context features
    # - Comment-submission overlap

    # To-Do: add user features
    # from Sammy's code; should be in DB

    def __init__(self, original, parent=None):
        """
        Basic constructor. Initializes references 
        to comment object (original) and parent,
        and sets other fields to None.
        """
        self.original = original
        self.parent = parent

        # Initialize temp vars as None
        for name in self.vars_temp:
            setattr(self, name, None)
        # Initialize features as None
        for name in self.vars_feature:
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


    def pos_tag(self, tagger=nltk.tag.pos_tag):
        """Calculate part-of-speech tags."""
        self.posTags = map(tagger, self.words)

    def calc_nouns_verbs(self):
        """Count all nouns and verbs, from Penn Treebank POS tags."""
        self.n_nouns = len([w for w,t in itertools.chain(*self.posTags)
                            if t.startswith("NN")])
        self.n_verbs = len([w for w,t in itertools.chain(*self.posTags)
                            if t.startswith("VB")])
