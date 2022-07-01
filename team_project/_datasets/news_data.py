import pathlib
import pandas as pd
import treform as ptm
import re
import pickle

####################
# ProQuest news
# - Format : Title                  |   Abstract            |   StoreId                     |   ArticleType |   Authors     |   companies               |   copyright       |
#            documentType           |   entryDate           |   issn                        |   issue       |   language    |   languageOfSummary       |   pages           |
#            placeOfPublication     |   pubdate             |   pubtitle                    |   year        |   volume      |   DocumentURL             |   classification  |
#            classificationCodes    |   identifierKeywords  |   majorClassificationCodes    |   notes       |   startPage   |   subjectClassifications  |   subjectTerms    |
#            subjects               |   URL                 |   FindACopy                   |   Database    |   coden       |   AlternateTitle          |   originalTitle   |
#            elecPubDate            |   full text
# - 기간 : 2008.01-2022.04
####################

def load_for_coword(target_index):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'news.xlsx'
    loc_stopwords = here / 'stopwordsEng.txt'

    # 데이터 로드
    df_news = pd.read_excel(loc_data)
    target = df_news.iloc[:, [target_index]].astype(str).values.tolist()
    # [[document1], ...] → [document1, ...]
    target = sum(target, [])

    # 전처리
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Word(),
                            ptm.tagger.NLTK(),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(1, 1),
                            ptm.helper.StopwordFilter(file=loc_stopwords))
    result = pipeline.processCorpus(target)

    # 구조 변경 : Sentence co-occurrence word를 찾기 위해 하나의 setence를 하나의 document로 변경
    documents = []
    for doc in result:
        for sent in doc:
            new_sent = ' '.join(sent)
            new_sent = re.sub('[^A-Za-z0-9가-힣_ ]+', '', new_sent)
            new_sent = new_sent.strip()
            if len(new_sent) > 0:
                documents.append(new_sent)
    return documents

def load_for_dmr(timestamp_index, target_index, timestamp_pattern='%Y', reuse_preproc=False):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'news.xlsx'
    loc_stopwords = here / 'stopwordsEng.txt'

    # 기전처리된 파일 사용 시
    if reuse_preproc:
        with open(here / 'news_pp_for_dmr.pkl', 'rb') as fin:
            timestamps = pickle.load(fin)
            documents = pickle.load(fin)
        fin.close()
        return timestamps, documents

    # 데이터 로드
    df_news = pd.read_excel(loc_data)
    timestamps = df_news.iloc[:, timestamp_index].dt.strftime(timestamp_pattern).tolist()
    target = df_news.iloc[:, [target_index]].astype(str).values.tolist()
    # [[document1], ...] → [document1, ...]
    target = sum(target, [])

    # 전처리
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Word(),
                            ptm.tagger.NLTK(),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),  # 품사 태크 제거
                            ptm.ngram.NGramTokenizer(1, 1),
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
    loc_data = here / 'news.xlsx'
    loc_stopwords = here / 'stopwordsEng.txt'

    # 기전처리된 파일 사용 시
    if reuse_preproc:
        with open(here / 'news_pp_for_bert.pkl', 'rb') as fin:
            timestamps = pickle.load(fin)
            documents = pickle.load(fin)
        fin.close()
        return timestamps, documents

    # 데이터 로드
    df_news = pd.read_excel(loc_data)
    timestamps = df_news.iloc[:, timestamp_index].dt.strftime(timestamp_pattern).tolist()
    target = df_news.iloc[:, [target_index]].astype(str).values.tolist()
    # [[document1], ...] → [document1, ...]
    target = sum(target, [])

    # 전처리
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Word(),
                            ptm.tagger.NLTK(),
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

# 정규식을 이용한 불필요한 문자열 제거(copyright, email 등)
def remove_stopword_with_reg_exp(target_column_name, reg_exp):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'news.xlsx'

    df_news = pd.read_excel(loc_data)
    df_news[target_column_name].replace(reg_exp, '', regex=True, inplace=True)
    df_news.to_excel('./news.xlsx', index=False, encoding='utf-8-sig')

# reg_exp1 = '(CREDIT|Credit): .+'
# reg_exp2 = '(CREDIT|Credit):.+'
# reg_exp3 = 'Word count:.+'
# remove_stopword_with_reg_exp(target_column_name='full text', reg_exp=reg_exp3)
