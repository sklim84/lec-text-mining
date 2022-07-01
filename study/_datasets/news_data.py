import pathlib
import treform as ptm
import pandas as pd
import re
import pickle

####################
# 뉴스 분석(네이버 뉴스)
# - Format : date | press | title | link | content
# - 기간 : 2017-2021
# - 총 8119건
# - 분석대상 : content
####################

####################
# 데이터 로드 및 전처리
# - 전처리 : Tokenizing, POS tagging & filtering, n-gram, Stopword Filtering
####################
def load_for_keyword(target):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'news.csv'
    loc_stopwords = here / 'stopwordsKor.txt'

    # 데이터 로드
    corpus = ptm.CorpusFromCSVFile(loc_data, target)
    # 전처리
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab('C:\\mecab\\mecab-ko-dic'),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(1, 1),
                            ptm.helper.StopwordFilter(file=loc_stopwords))
    result = pipeline.processCorpus(corpus.docs[1:])

    documents = []
    for doc in result:
        document = ''
        for sent in doc:
            new_sent = ' '.join(sent)
            new_sent = re.sub('[^A-Za-z0-9가-힣_ ]+', '', new_sent)
            new_sent = new_sent.strip()
            document += new_sent
        documents.append(document)

    return documents

####################
# Term Weighting을 위한 데이터 로드 및 처리
# - 전처리 : Tokenizing, POS tagging & filtering, Stopword Filtering
####################
def load_for_term_weighting(label_index, target_index):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'news.csv'
    loc_stopwords = here / 'stopwordsKor.txt'

    # 데이터 로드
    label = ptm.CorpusFromCSVFile(loc_data, label_index)
    content = ptm.CorpusFromCSVFile(loc_data, target_index)
    # 전처리
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab('C:\\mecab\\mecab-ko-dic'),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),  # 품사 태크 제거
                            ptm.helper.StopwordFilter(file=loc_stopwords))
    return label.docs[1:], pipeline.processCorpus(content.docs[1:])

####################
# Term Burstiness를 위한 데이터 로드 및 처리
# - 전처리 : Tokenizing, POS tagging & filtering, n-gram, Stopword Filtering
####################
def load_for_term_burstiness(target_index, ngram_min, ngram_max):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'news.csv'
    loc_stopwords = here / 'stopwordsKor.txt'

    # 데이터 로드
    df = pd.read_csv(loc_data)
    # 전처리
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab('C:\\mecab\\mecab-ko-dic'),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),  # 품사 태크 제거
                            ptm.ngram.NGramTokenizer(ngram_min, ngram_max),
                            ptm.helper.StopwordFilter(file=loc_stopwords))
    result = pipeline.processCorpus(df.iloc[:, target_index])

    # 데이터 구조 변경 : documents/sentences/words -> documents/words
    documents = []
    for doc in result:
        document = ''
        for sent in doc:
            new_sent = ' '.join(sent)
            new_sent = re.sub('[^A-Za-z0-9가-힣_ ]+', '', new_sent)
            new_sent = new_sent.strip()
            document += ' ' + new_sent
        documents.append(document)

    # 전처리 결과 반영
    print(documents)
    df['content'] = documents
    return df

def load_for_dmr(timestamp_index, target_index, timestamp_pattern='%Y', reuse_preproc=False):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'news.csv'
    loc_stopwords = here / 'stopwordsKor.txt'

    # 기전처리된 파일 사용 시
    if reuse_preproc:
        with open(here / 'news_pp_for_dmr.pkl', 'rb') as fin:
            timestamps = pickle.load(fin)
            documents = pickle.load(fin)
        fin.close()
        return timestamps, documents

    # 데이터 로드
    df_news = pd.read_csv(loc_data)
    df_news.iloc[:, timestamp_index] = pd.to_datetime(df_news.iloc[:, timestamp_index], errors='coerce')
    timestamps = df_news.iloc[:, timestamp_index].dt.strftime(timestamp_pattern)
    target = df_news.iloc[:, [target_index]].astype(str).values.tolist()
    # [[document1], ...] → [document1, ...]
    target = sum(target, [])

    # 전처리
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab('C:\\mecab\\mecab-ko-dic'),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),  # 품사 태크 제거
                            ptm.helper.StopwordFilter(file=loc_stopwords))
    result = pipeline.processCorpus(target)

    documents = []
    for doc in result:
        document = []
        for sent in doc:
            for word in sent:
                new_word = re.sub('[^A-Za-z0-9가-힣_ ]+', '', word)
                new_word = new_word.strip()
                if len(new_word) > 0:
                    document.append(new_word)
        documents.append(document)

    # 전처리된 결과 저장
    with open(here / 'news_pp_for_dmr.pkl', 'wb') as fout:
        pickle.dump(timestamps, fout)
        pickle.dump(documents, fout)
    fout.close()

    return timestamps, documents


def load_for_bert(timestamp_index, target_index, timestamp_pattern='%Y', reuse_preproc=False):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'news.csv'
    loc_stopwords = here / 'stopwordsKor.txt'

    # 기전처리된 파일 사용 시
    if reuse_preproc:
        with open(here / 'news_pp_for_bert.pkl', 'rb') as fin:
            timestamps = pickle.load(fin)
            documents = pickle.load(fin)
        fin.close()
        return timestamps, documents

    df_news = pd.read_csv(loc_data)
    df_news.iloc[:, timestamp_index] = pd.to_datetime(df_news.iloc[:, timestamp_index], errors='coerce')
    timestamps = df_news.iloc[:, timestamp_index].dt.strftime(timestamp_pattern).tolist()
    target = df_news.iloc[:, [target_index]].astype(str).values.tolist()
    # [[document1], ...] → [document1, ...]
    target = sum(target, [])

    # 전처리
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab('C:\\mecab\\mecab-ko-dic'),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),  # 품사 태크 제거
                            ptm.helper.StopwordFilter(file=loc_stopwords))
    result = pipeline.processCorpus(target)

    documents = []
    for doc in result:
        document = ''
        for sent in doc:
            for word in sent:
                new_word = re.sub('[^A-Za-z0-9가-힣_ ]+', '', word)
                new_word = new_word.strip()
                if len(new_word) > 0:
                    document = ' '.join([document, new_word])
        documents.append(document)

    # 전처리된 결과 저장
    with open(here / 'news_pp_for_bert.pkl', 'wb') as fout:
        pickle.dump(timestamps, fout)
        pickle.dump(documents, fout)
    fout.close()

    return timestamps, documents

load_for_bert(0, 4)