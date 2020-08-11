### Classification Benchmarks for Under-resourced Bengali Language Based on Multichannel Convolutional-LSTM Network
Code and supplementary materials for our paper titled "Classification Benchmarks for Under-resourced Bengali Language based on Multichannel Convolutional-LSTM Network", taccepted as a full paper at 7th IEEE International Conference on Data Science and Advanced Analytics (IEEE DSAA,2020). 

#### Methods
Exponential growths of social media and micro-blogging sites not only provide platforms for empowering freedom of expressions and individual voices but also enables people to express anti-social behaviour like online harassment, cyberbullying, and hate speech. Numerous works have been proposed to utilize these data for social and anti-social behaviours analysis, document characterization, and sentiment analysis by predicting the contexts mostly for highly resourced languages such as English. However, there are languages that are under-resources, e.g., South Asian languages like Bengali, Tamil, Assamese, Telugu that lack of computational resources for the (NLP) tasks. 

In this paper, we provide several classification benchmarks for Bengali, an under-resourced language. We prepared three datasets of expressing hate, commonly used topics, and opinions for hate speech detection, document classification, and sentiment analysis, respectively. We built the largest Bengali word embedding models to date based on 250 million articles, which we call 'BengFastText'. We perform three different experiments, covering document classification, sentiment analysis, and hate speech detection. We incorporate word embeddings into a Multichannel Convolutional-LSTM (MConv-LSTM) network for predicting different types of hate speech, document classification, and sentiment analysis. Experiments demonstrate that 'BengFastText' can capture the semantics of words from respective contexts correctly. Evaluations against several baseline embedding models, e.g., Word2Vec and GloVe yield up to 92.30%, 82.25%, and 90.45% F1-scores in case of document classification, sentiment analysis, and hate speech detection, respectively during 5-fold cross-validation tests.

#### Requirements
* nltk >= 1.11.1
* regex >= 2016.6.24
* lxml >= 3.3.3
* numpy >= 1.11.2
	
#### References
* Check [this](https://github.com/Kyubyong/wordvectors) to know how to prepare fastText model for other languages such as Korean, Japanese, French etc.

#### Workflow of training the BengFastText model 
* STEP 1. Download [this](https://drive.google.com/open?id=199Z3KlTLvoApGixb5rwDIkbsmsl2n56F) raw Bengali texts in case you don't want to collect all the raw text.
* STEP 2. Extract running texts to `data/` folder.
* STEP 3. Run `build_corpus.py` using `python3 build_corpus.py`
* STEP 4. Run `sudo ./make_wordvector.sh` to get the BengFastText word vectors but make sure it's already executable.

#### Pre-trained BengFastText model
| Language  |  UTF-8 | Vector Size | Corpus Size  | Vocabulary Size | 
| ---       |---        |---           |---           |---           |
|[Bengali (BengFastText)](https://drive.google.com/open?id=1Q_45PQpRWQvZL2p8sIngmgg6Tr5YbKmH) \| [Bengali (f)](https://drive.google.com/open?id=1Q_45PQpRWQvZL2p8sIngmgg6Tr5YbKmH)|bn|300|250M |30059| negative sampling |

#### Citation request
If you use the code of this repository in your research, please consider citing the folowing papers:

    @inproceedings{karim2020BengaliNLP,
        title={Classification Benchmarks for Under-resourced Bengali Language based on Multichannel Convolutional-LSTM Network},
        author={Md. Rezaul Karim, Bharathi Raja Chakravarti, John P. McCrae, and Michael Cochez},
        conference={7th IEEE International Conference on Data Science and Advanced Analytics (IEEE DSAA,2020)},
        year={2020}
    }

#### Contributing
For any questions, feel free to open an issue or contact at rezaul.karim@rwth-aachen.de
