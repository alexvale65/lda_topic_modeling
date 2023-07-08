import re
import pickle
import json
import logging
import warnings
import pandas as pd
import itertools

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from gensim.models import Phrases
from gensim.models.phrases import Phraser

# spacy for lemmatization
import spacy

from nltk.corpus import stopwords


from pathlib import Path
import os
from os import listdir
from os.path import isfile, join


DATA_FOLDER = Path(os.path.dirname(__file__))

logfile = DATA_FOLDER / 'logs' / 'output'/ 'model' /f"{pd.to_datetime('today').strftime('%b %d %Y %I:%M%p')}.log"
logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    filemode="w",
                    level=logging.INFO,
                    filename=logfile)
logger = logging.getLogger()

warnings.filterwarnings("ignore")

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def scopus_journal_metrics():
    print('adding scopus jounal metrics')
    path_metric = DATA_FOLDER / 'input'/ 'CiteScore2020.xlsx'
    df_scopus_metric = pd.read_excel(path_metric
                                     , sheet_name='CiteScore 2020'
                                     , usecols=['Title', 'CiteScore 2020', 'SNIP', 'SJR'])
    df_scopus_metric = df_scopus_metric.rename(columns={'Title': 'Journal'})
    df_scopus_metric = df_scopus_metric.drop_duplicates(subset=['Journal'])
    return df_scopus_metric


def _corpus(path_corpus, reload=True):
    if reload:
        print('reloading corpus dataset')
        path_raw_corpus = DATA_FOLDER / 'input' / 'corpus'
        year_slice_files = [f for f in listdir(path_raw_corpus) if isfile(join(path_raw_corpus, f))]
        df_corpus = pd.concat([pd.read_csv(path_raw_corpus + f) for f in year_slice_files]).drop_duplicates()
        df_corpus.to_csv(path_corpus, index=False)
    else:
        print('loading corpus by backup')
        df_corpus = pd.read_csv(path_corpus)
    df_scopus_metric = scopus_journal_metrics()
    return df_corpus.merge(df_scopus_metric
                           , how='left'
                           , left_on='Source title'
                           , right_on='Journal')

def load_data(variable_fields):
    path_corpus = DATA_FOLDER / 'output' / 'data' / 'corpus.csv'
    df_corpus = _corpus(path_corpus)
    if isinstance(variable_fields, list) and len(variable_fields) > 0:
        df_corpus[variable_fields] = df_corpus[variable_fields].fillna(' ')
        df_corpus[variable_fields[0]] = df_corpus[variable_fields[0]].map(lambda x: x + ' ')
        df_corpus['variable'] = df_corpus[variable_fields].sum(axis=1)
    return df_corpus


def data_cleaning(df_corpus_raw, treshold_tokens):
    print('start cleaning process')
    print(f'-raw publications database: {df_corpus_raw.shape[0]}')
    # field availables
    df_corpus_raw = df_corpus_raw[~df_corpus_raw['DOI'].isnull()]
    df_corpus_raw = df_corpus_raw[df_corpus_raw['Authors'].str.strip() != '[No author name available]']
    df_corpus_raw = df_corpus_raw[df_corpus_raw['Abstract'].str.strip() != '[No abstract available]']
    print(f'-publications with DOI|Authors|Abstract: {df_corpus_raw.shape[0]}')
    # remove duplicated
    df_corpus_raw = df_corpus_raw.drop_duplicates(subset=['Title', 'DOI'])
    print(f'-drop duplicated by Title|DOI: {df_corpus_raw.shape[0]}')
    # filter publications by trheshold number of tokens in the absrtact
    df_corpus_raw['ntokens'] = df_corpus_raw['Abstract'].apply(lambda x: len(x.split(' ')))
    treshold = int(df_corpus_raw['ntokens'].quantile(treshold_tokens))
    df_corpus_raw = df_corpus_raw[df_corpus_raw['ntokens'] >= treshold]
    print(f"-applying treshold for number of tokens ({treshold}): ", df_corpus_raw.shape[0])
    df_corpus_raw = df_corpus_raw.reset_index(drop=True)
    df_corpus_raw['No_Document'] = df_corpus_raw.index
    # lower case
    df_corpus_raw['variable'] = df_corpus_raw['variable'].map(lambda x: x.strip().lower())
    # Remove new line characters
    df_corpus_raw['variable'] = df_corpus_raw['variable'].map(lambda x: re.sub('\s+', ' ', x))
    # Remove distracting single quotes
    df_corpus_raw['variable'] = df_corpus_raw['variable'].map(lambda x: re.sub("\'", "", x))
    return df_corpus_raw


def preprocessing_textmining_field(string, extend_stop_words):
    raw = ''.join(string).split(' ')
    stp1 = list(sent_to_words(raw, min_len=3, max_len=30))
    stp2 = remove_stopwords(texts=stp1, extend_stop_words=extend_stop_words)
    stp3 = [' '.join(i) for i in stp2 if len(i) > 0]
    lemm = [[token.lemma_ for token in nlp(sent) if token.pos_ in ['NOUN', 'VERB', 'ADJ']] for sent in stp3]
    tmp_lemm = [' '.join(i) for i in lemm if len(i) > 0]
    return ' '.join(tmp_lemm)


def training_ngrams(df_data, field):
    all_text = df_data[field].sum().split(' ')
    bigram = Phrases([all_text], min_count=5)
    bigram_phraser = Phraser(bigram)
    bigram_lemmatized = bigram_phraser[all_text]
    trigram = Phrases([bigram_lemmatized], min_count=5)
    trigram_phraser = Phraser(trigram)
    trigram_lemmatized = trigram_phraser[bigram_lemmatized]
    return [' '.join(lemma.split('_')) for lemma in trigram_lemmatized if '_' in lemma]


def replace_ngrams(string, ngram_list):
    for ng in ngram_list:
        string = string.replace(ng, ng.replace(' ', '_'))
    return string


def sent_to_words(sentences, min_len=3, max_len=30):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(simple_preprocess(str(sentence), deacc=True, min_len=min_len, max_len=max_len))


def remove_stopwords(texts, extend_stop_words, len_max=15, len_min=3):
    stop_words = stopwords.words('english')
    stop_words.extend(extend_stop_words)
    tokens = [simple_preprocess(str(doc), min_len=len_min, max_len=len_max) for doc in texts]
    return [[word for word in token if word not in stop_words] for token in tokens]


def pre_processing(variable_fields, extend_stop_words, treshold_tokens):
    raw_data = load_data(variable_fields=variable_fields)
    if isinstance(variable_fields, list) and len(variable_fields) > 0:
        print(f'using variables: ', variable_fields)
        data = data_cleaning(df_corpus_raw=raw_data, treshold_tokens=treshold_tokens)
        print('pre processing textmining field')
        print('-remove stopwords')
        print('-lemmatization')
        data['variable_processed'] = data['variable'].map(lambda x: preprocessing_textmining_field(x
                                                                                                   , extend_stop_words)
                                                          )
        print('-assenbling Ngrams')
        ngrams = training_ngrams(data, 'variable_processed')
        print('-replacing tokens by Ngrams')
        data['variable_lemmatized'] = data['variable_processed'].apply(lambda x: replace_ngrams(x, ngrams))
        print('recording backup')
        data.to_csv(DATA_FOLDER / 'input' / 'corpus' / 'preProcessedCorpus.csv', index=False)
        raw_data.to_csv(DATA_FOLDER / 'corpus' / 'RawCorpus.csv', index=False)
        with open(DATA_FOLDER / 'input' / 'corpus' /'generalNgrams.json', 'w') as f:
            json.dump(ngrams, f)
        print('end process')
        return raw_data, data, ngrams
    print('enter variable fields to starting preprocessing')


def load_model_data(variable_fields, treshold_tokens=0.25, use_backup=False, extend_stop_words=[]):
    if use_backup:
        data = pd.read_csv(DATA_FOLDER / 'input' / 'corpus' / 'preProcessedCorpus.csv')
        raw_data = pd.read_csv(DATA_FOLDER / 'input' / 'corpus' / 'RawCorpus.csv')
        with open(DATA_FOLDER / 'input' / 'corpus' /'generalNgrams.json') as json_file:
            ngrams = json.load(json_file)
    else:
        raw_data, data, ngrams = pre_processing(variable_fields=variable_fields
                                                , extend_stop_words=extend_stop_words
                                                , treshold_tokens=treshold_tokens)


    sentences = data['variable_lemmatized'].values.tolist()
    vector_words = list(sent_to_words(sentences))
    # Create Dictionary
    id2word = corpora.Dictionary(vector_words)
    # Create Corpus
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in vector_words]
    return raw_data, data, corpus, id2word, ngrams


def LDA_model(corpus, id2word, texts, num_keywords, num_topics, training=True):
    LDA_models = {}
    LDA_topics = {}
    coherences = []
    coherences_u_mass = []
    perplexities = []
    base_path = DATA_FOLDER / 'output' / 'models'
    for i in num_topics:
        print(f"topic #{i}")
        filepath = base_path / f'lda_model_{i}.pk'
        if training or os.path.isfile(filepath) is False:
            print('starting training model')
            model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=i,
                                                    update_every=1,
                                                    chunksize=len(corpus),
                                                    passes=20,
                                                    alpha='auto',
                                                    random_state=42)
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        elif training is False and os.path.isfile(filepath):
            print('using backup training model')
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        else:
            print('interrupt process for lack information')
        shown_topics = model.show_topics(num_topics=i, num_words=num_keywords, formatted=False)
        LDA_topics[i] = [[word[0] for word in topic[1]] for topic in shown_topics]
        LDA_models[i] = model
        coherences.append(CoherenceModel(model=model
                                         , texts=texts
                                         , dictionary=id2word
                                         , coherence='c_v').get_coherence())
        coherences_u_mass.append(CoherenceModel(model=model
                                                , texts=texts
                                                , dictionary=id2word
                                                , coherence='u_mass').get_coherence())
        perplexities.append(model.log_perplexity(corpus))
    return LDA_models, LDA_topics, coherences, coherences_u_mass, perplexities

def format_topics_sentences(ldamodel, corpus, concat=None):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        if concat and isinstance(concat, tuple):
            row = concat_topic(row, from_topic=concat[1], to_topic=concat[0])
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num, topn=20)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords])
                                                       , ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    return sent_topics_df

def concat_topic(x, from_topic, to_topic):
    adjust_tuple = []
    for t in x:
        k = to_topic if t[0] in from_topic else t[0]
        adjust_tuple.append((k, t[1]))
    new_tuple = [(key, sum(num for _, num in value)) for key, value in itertools.groupby(adjust_tuple, lambda x: x[0])]
    return new_tuple


if __name__ == '__main__':
    ext_stop_words = []
    raw_data, data, corpus, id2word, ngrams = load_model_data(variable_fields=['Title', 'Abstract']
                                                              , use_backup=True
                                                              , extend_stop_words=ext_stop_words)
    num_topics = list(range(20)[2:])
    num_keywords = 20
    texts = [text.split(' ') for text in data['variable_lemmatized'].tolist()]
    LDA_models, LDA_topics, coherences, coherences_u_mass, perplexities = LDA_model(corpus=corpus
                                                                                      , id2word=id2word
                                                                                      , texts=texts
                                                                                      , num_topics=num_topics
                                                                                      , num_keywords=num_keywords
                                                                                      , training=True)

