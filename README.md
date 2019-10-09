### A Linguistic Analytics Framework for Under-Resource Languages Based on Multichannel Convolutional-LSTM Network
Code and supplementary materials for our paper titled "A Linguistic Analytics Framework for Under-Resource Languages Based on Multichannel Convolutional-LSTM Network", submitted to Natural Language Engineering journal (Cambridge) [2018 Impact Factor: 1.130]. 

#### Methods
Exponential growths of social media and micro-blogging sites not only provides platformsfor empowering freedom of expressions and individual voices but also enables people toexpress anti-social behavior like online harassment, cyberbullying, and hate speech. Nu-merous works have been proposed to utilize these data for social and anti-social behavioursanalysis, document characterization,  and  sentiment analysis by predicting the contextsmostly for highly resourced languages such as English. However, there are languages thatare  under-resources, e.g., South Asian languages like  Bengali, Tamil, Assamese, Teluguthat lack of computational resources for the natural language processing (NLP) tasks. 

In this paper, we propose a linguistic analytics framework for Bengali, anunder-resourcedlanguage. Our framework  consists of annotated datasets and word embeddings. Threedatasets were collected and annotated expressing hate, commonly used topics, and opin-ions for hate speech detection, document classification, and sentiment analysis,  respec-tively. As part of the framework, we built the largest Bengali word embedding models todate based on 250 million articles, which we callBengFastText. To evaluate our frame-work, we perform experiments on document classification, sentiment analysis, and hatespeech detection for Bengali. To this end, we incorporate word embeddings into a Multi-channel Convolutional-LSTM (MConv-LSTM) network for predicting different types of hatespeech, document classification, and sentiment analysis. Experiments demonstrate that BengFastText can capture the semantics of words from respective contexts correctly. Eval-uations against several baseline embedding models, e.g., Word2Vec and GloVe yield upto 92.30%, 82.25%, and 90.45% F1-scores in case of document classification, sentimentanalysis, and hate speech detection, respectively during 5-fold cross-validation tests. 

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

#### Pre-trained BengWord2Vec model
| Language  |  UTF-8 | Vector Size | Corpus Size  | Vocabulary Size | 
| ---       |---        |---           |---           |---           |
|[Bengali (BengFastText)](https://drive.google.com/open?id=1Q_45PQpRWQvZL2p8sIngmgg6Tr5YbKmH) \| [Bengali (f)](https://drive.google.com/open?id=1Q_45PQpRWQvZL2p8sIngmgg6Tr5YbKmH)|bn|300|250M |30059| negative sampling |

* Check [this](https://drive.google.com/open?id=1Q_45PQpRWQvZL2p8sIngmgg6Tr5YbKmH) for the pre-trained BengFastText model.


#### Citation request
If you use the code of this repository in your research, please consider citing the folowing papers:

    @inproceedings{karim2019XAI,
        title={A Linguistic Analytics Framework for Under-Resource Languages Based on Multichannel Convolutional-LSTM Network},
        author={Md. Rezaul Karim, Bharathi Raja Chakravarti, Mihael Arcan, John P. McCrae, and Michael Cochez},
        journal={Natural Language Engineering},
        year={2019}
    }

#### Contributing
For any questions, feel free to open an issue or contact at rezaul.karim@rwth-aachen.de

