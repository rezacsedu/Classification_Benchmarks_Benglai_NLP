# BengFastText model and it's applications

Our BengFastText is technically a fastText model, which is based on 250 million articles. More complete computational resurces and interactive notebooks will be added in upcoimng days. 

## Requirements
* nltk >= 1.11.1
* regex >= 2016.6.24
* lxml >= 3.3.3
* numpy >= 1.11.2
	
## References
* Check [this](https://github.com/Kyubyong/wordvectors) to know how to prepare fastText model for other languages such as Korean, Japanese, French etc.

## Workflow of training the BengFastText model 
* STEP 1. Download [this](https://drive.google.com/open?id=199Z3KlTLvoApGixb5rwDIkbsmsl2n56F) raw Bengali texts in case you don't want to collect all the raw text.
* STEP 2. Extract running texts to `data/` folder.
* STEP 3. Run `build_corpus.py` using `python3 build_corpus.py`
* STEP 4. Run `sudo ./make_wordvector.sh` to get the BengFastText word vectors but make sure it's already executable.

## Pre-trained BengWord2Vec model
| Language  |  UTF-8 | Vector Size | Corpus Size  | Vocabulary Size | 
| ---       |---        |---           |---           |---           |
|[Bengali (BengFastText)](https://drive.google.com/open?id=1Q_45PQpRWQvZL2p8sIngmgg6Tr5YbKmH) \| [Bengali (f)](https://drive.google.com/open?id=1Q_45PQpRWQvZL2p8sIngmgg6Tr5YbKmH)|bn|300|250M |30059| negative sampling |

* Check [this](https://drive.google.com/open?id=1Q_45PQpRWQvZL2p8sIngmgg6Tr5YbKmH) for the pre-trained BengFastText model.


