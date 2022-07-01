import treform as ptm
import pathlib


def load_for_train():
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / '3_class_naver_news.csv'
    loc_stopwords = here / 'stopwordsKor.txt'

    mecab_path = 'C:\\mecab\\mecab-ko-dic'
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab(mecab_path),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(2, 2),
                            ptm.helper.StopwordFilter(file=loc_stopwords))

    corpus = ptm.CorpusFromFieldDelimitedFileForClassification(loc_data, delimiter=',', doc_index=4, class_index=1,
                                                               title_index=3)

    tups = corpus.pair_map
    class_list = []
    for id in tups:
        class_list.append(tups[id])

    result = pipeline.processCorpus(corpus)
    documents = []
    for doc in result:
        document = ''
        for sent in doc:
            document += " ".join(sent)
        documents.append(document)

    return documents, class_list

def load_for_predict():
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'navernews.txt'
    loc_stopwords = here / 'stopwordsKor.txt'

    mecab_path = 'C:\\mecab\\mecab-ko-dic'
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab(mecab_path),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(2, 2),
                            ptm.helper.StopwordFilter(file=loc_stopwords))

    corpus = ptm.CorpusFromFieldDelimitedFile(loc_data, 3)

    result = pipeline.processCorpus(corpus)
    documents = []
    for doc in result:
        document = ''
        for sent in doc:
            document += " ".join(sent)
        documents.append(document)

    return documents