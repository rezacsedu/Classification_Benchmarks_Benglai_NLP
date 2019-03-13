# BengWord2Vec model and it's applications

Our BengWord2Vec is technically a Word2Vec model, which is based on 250 articles. This is based on https://github.com/Kyubyong/wordvectors, implementation but with additional midifications and updates. 

## Requirements
* nltk >= 1.11.1
* regex >= 2016.6.24
* lxml >= 3.3.3
* numpy >= 1.11.2
* gensim > =0.13.1 (for Word2Vec)
	
## References
* Check [this](https://github.com/Kyubyong/wordvectors) to know how to prepare Word2Vec model for other languages such as Korean, Japanese, French etc.

## Workflow of training the BengWord2Vec model 
* STEP 1. Download the [this](https://dumps.wikimedia.org/backup-index.html) in case you don't want to collect all the raw text.
* STEP 2. Extract running texts to `data/` folder.
* STEP 3. Run `build_corpus.py` using `python3 build_corpus.py`
* STEP 4. Run `sudo ./make_wordvector.sh` to get the BengWord2Vec word vectors but make sure it's already executable.

## Pre-trained BengWord2Vec model
| Language  |  UTF-8 | Vector Size | Corpus Size  | Vocabulary Size | 
| ---       |---        |---           |---           |---           |
|[Bengali (BengWord2Vec)](https://drive.google.com/open?id=1Q_45PQpRWQvZL2p8sIngmgg6Tr5YbKmH) \| [Bengali (f)](https://drive.google.com/open?id=1Q_45PQpRWQvZL2p8sIngmgg6Tr5YbKmH)|bn|300|250M |30059| negative sampling |

* Check [this](https://drive.google.com/open?id=1Q_45PQpRWQvZL2p8sIngmgg6Tr5YbKmH) for the pre-trained BengWord2Vec model.


