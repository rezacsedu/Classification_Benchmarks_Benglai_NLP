#!/usr/bin/python2
import nltk
import os
import codecs
import argparse
import numpy as np

import sys
reload(sys)
sys.setdefaultencoding("ISO-8859-1")

import gensim # In case you have difficulties installing gensim, you need to consider installing conda.

model = gensim.models.KeyedVectors.load_word2vec_format('/home/karim/wordvectors/data/bangla_word2vec_model/bn.bin', binary=True)

# Some predefined functions that show content related information for given words
print(model.most_similar(positive=['woman', 'king'], negative=['man']))

print(model.doesnt_match("breakfast cereal dinner lunch".split()))

print(model.similarity('woman', 'man'))