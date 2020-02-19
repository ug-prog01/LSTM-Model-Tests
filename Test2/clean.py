# import pandas as pd
#AUTHOR: UTKARSH MISHRA
import re

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')


"""This function removes the punctuation marks from the given transcript and returns the processed
clean text which is then fed to the model in the following function to give the resulting output.
The regex expression can be modified to fit the type of transcript generated.
"""

def clean_text(sents):
    """
    Cleaning File
    """
    ret_sents = []
    for sent in sents:
        text = sent.lower()
        text = BAD_SYMBOLS_RE.sub('', text)
        ret_sents.append(text)
    return ret_sents

def clean_text_one(sents):
    """
    Cleaning Single Sentence
    """
    text = sents.lower()
    text = BAD_SYMBOLS_RE.sub('', text)
    return text