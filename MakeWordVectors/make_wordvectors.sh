#!/bin/bash

#### Set your hyper-parameters here ####
############## START ###################
lcode="bn" # ISO 639-1 code of target language. See `lcodes.txt`.
max_corpus_size=1000000000 # the maximum size of the corpus. Feel free to adjust it according to your computing power.
vector_size=300 # the size of a word vector
window_size=5 # the maximum distance between the current and predicted word within a sentence.
vocab_size=2000000 # the maximum vocabulary size
num_negative=5 # the int for negative specifies how many “noise words” should be drawn
############## END #####################

echo "Making wordvectors"
python make_wordvectors.py --lcode=${lcode} --vector_size=${vector_size} --window_size=${window_size} --vocab_size=${vocab_size} --num_negative=${num_negative}

