##
# Social Web Comment Ranking
# CS224U Spring 2014
# Stanford University 
#
# Feature Extraction
# Text Preprocessing Methods
#
# Ian F. Tenney
# Narek Tovmasyan
# June 7, 2014
##

import re
import nltk

# Tokenizers for input
sent_tokenize = nltk.tokenize.sent_tokenize
word_tokenize = nltk.tokenize.word_tokenize
stemmer = nltk.stem.porter.PorterStemmer()


"""
A simple routine for parsing the comment text. 
Strips out the reddit martdown, as well as categorizes 
some useful tokens.
"""
def preprocessText(text):
    # p = re.compile(r'\[(.*?)\]\(.*\)')
    r = re.compile(r'\[(.*?)\]\(.*?\)')
    # text =  r.sub(r"URL_TAG(\1)", text)
    test = r.sub(r"\1 [URL_TAG]")

    r_link = re.compile(r'[0-9a-z]+:[/]+[^ ]+')
    text = r_link.sub("[LINK_TAG]", text)

    # For removing markdown for bolding the text..
    r_bold = re.compile(r'\*\*(.*?)\*\*')
    text = r_bold.sub(r'[EMPHASIS] \1', text)

    # Could be even more general (stripping both bolded & italicized..)
    r_bi = re.compile(r'\*\*?(.*?)\*?\*')
    text =  r_bi.sub(r'\1', text) 

    r_num = re.compile(r'(-?[0-9]+\.?[0-9]*[A-z]*)')
    text = r_num.sub(r'[NUM_TAG]', text)

    r_etal = re.compile("et al.")
    text = r_etal.sub("[ETALIA]", text)

    r_quot = re.compile("&gt;")
    text = r_quot.sub("[QUOTE]", text)

    return text

def wordfilter(word):
    """Crude filter to skip punctuation and URLs
    Note that this leaves in the 'http' token, which
    can still be used as a proxy for URLs."""
    if len(word) < 2: return False
    if "www" in word: return False
    # if not a.isalpha(): return False
    return True

def default_tokenizer(text, wf=wordfilter, pre=None):
    sentences = sent_tokenize(text)
    words = (word_tokenize(s) for s in sentences) # lazy generator
    # To-Do: make this filtering more robust, add special tokens / transform
    words_filtered = (w for w in lazy_chain(words) if wf(w)) # lazy generator
    # words_filtered = (w for w in itertools.chain(*words) if wf(w)) # lazy generator
    return [stemmer.stem(w) for w in words_filtered]
