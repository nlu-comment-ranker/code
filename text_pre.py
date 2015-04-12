##
# Social Web Comment Ranking
#
# Feature Extraction
# Text Preprocessing Methods
##

import re
import nltk

# Like itertools.chain, but doesn't 
# require eager argument expansion
# at the top level
def lazy_chain(listgen):
    for l in listgen:
        for e in l:
            yield e

# Tokenizers for input
sent_tokenize = nltk.tokenize.sent_tokenize
word_tokenize = nltk.tokenize.word_tokenize
stemmer = nltk.stem.porter.PorterStemmer()

def skipping_stemmer(word):
    """Special wrapper to have stemmer ignore special tokens."""
    if word.startswith("__"): return word
    else: return stemmer.stem(word)

"""
A simple routine for parsing the comment text. 
Strips out the reddit martdown, as well as categorizes 
some useful tokens.
"""
r_url = re.compile(r'\[(.*)\]\(.*\)')
r_link = re.compile(r'[0-9a-z]+:[/]+[^()\[\]{} ]+')
r_bold = re.compile(r'\*\*(.*?)\*\*')
r_ital = re.compile(r'\*(.*?)\*')
r_num = re.compile(r'(-?[0-9]+\.?[0-9]*[A-z]*)')
r_etal = re.compile("et al.")
r_quot = re.compile("&gt;")

def preprocessText(text):
    # p = re.compile(r'\[(.*?)\]\(.*\)')
    # r = re.compile(r'\[(.*?)\]\(.*?\)')
    # text =  r.sub(r"URL_TAG(\1)", text)
    text = r_url.sub(r"\1 __URL_TAG__", text)

    text = r_link.sub("__LINK_TAG__", text)

    # For removing markdown for bolding the text..
    text = r_bold.sub(r'__BOLD__ \1', text)

    # Could be even more general (stripping both bolded & italicized..)
    # r_bi = re.compile(r'\*\*?(.*?)\*?\*')
    # text =  r_bi.sub(r'\1', text) 
    text = r_ital.sub(r'__ITAL__ \1', text)

    text = r_num.sub(r'__NUM_TAG__', text)

    text = r_etal.sub("__ETALIA__", text)

    text = r_quot.sub("__QUOTE__", text)

    return text

def wordfilter(word):
    """Crude filter to skip punctuation and URLs
    Note that this leaves in the 'http' token, which
    can still be used as a proxy for URLs."""
    if len(word) < 2: return False
    if "www" in word: return False
    # if not a.isalpha(): return False
    return True

def default_tokenizer(text, pre=preprocessText, wf=wordfilter, stem=skipping_stemmer):
    if pre: text = pre(text) # preprocess -> special tokens, remove markdown
    sentences = sent_tokenize(text)
    words = (word_tokenize(s) for s in sentences) # lazy generator
    # To-Do: make this filtering more robust, add special tokens / transform
    words_filtered = (w for w in lazy_chain(words) if wf(w)) # lazy generator
    # words_filtered = (w for w in itertools.chain(*words) if wf(w)) # lazy generator
    return [stem(w) for w in words_filtered]
