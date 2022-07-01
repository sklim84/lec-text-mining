import pathlib
import pandas as pd
import pickle
import treform as ptm
import re

####################
# Scopus papers
# - Format : eid | cover_date(yyyy-mm-dd) | title | abstract | keywords
# - 기간 : 2008.01-2022.04
####################

def load_for_coword(target_index):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'papers.xlsx'
    loc_stopwords = here / 'stopwordsEng.txt'

    # 데이터 로드
    df_papers = pd.read_excel(loc_data)
    target = df_papers.iloc[:, [target_index]].astype(str).values.tolist()
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
    loc_data = here / 'papers.xlsx'
    loc_stopwords = here / 'stopwordsEng.txt'

    # 기전처리된 파일 사용 시
    if reuse_preproc:
        with open(here / 'papers_pp_for_dmr.pkl', 'rb') as fin:
            timestamps = pickle.load(fin)
            documents = pickle.load(fin)
        fin.close()
        return timestamps, documents

    # 데이터 로드
    df_papers = pd.read_excel(loc_data)
    timestamps = df_papers.iloc[:, timestamp_index].dt.strftime(timestamp_pattern).tolist()
    target = df_papers.iloc[:, [target_index]].astype(str).values.tolist()
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

    new_result = []
    # remove empty document
    for index, doc in enumerate(result):
        is_empty_doc = True
        for sent in doc:
            if len(sent) != 0:
                is_empty_doc = False

        if is_empty_doc:
            print(str(index) + ' document is empty')
            print('remove timestamp: {}'.format(timestamps[index]))
            timestamps[index] = -1
        else:
            new_result.append(doc)
    # remove timestamp related to empty document
    timestamps = [timestamp for timestamp in timestamps if timestamp != -1]

    documents = []
    for doc in new_result:
        document = []
        for sent in doc:
            for word in sent:
                new_word = re.sub('[^A-Za-z0-9가-힣_ ]+', '', word)
                new_word = new_word.strip()
                if len(new_word) > 0:
                    document.append(new_word)
        documents.append(document)

    # 전처리된 결과 저장
    with open(here / 'papers_pp_for_dmr.pkl', 'wb') as fout:
        pickle.dump(timestamps, fout)
        pickle.dump(documents, fout)
    fout.close()

    return timestamps, documents

####################
# Term Burstiness를 위한 데이터 로드 및 처리
# - 전처리 : Tokenizing, POS tagging & filtering, n-gram, Stopword Filtering
####################
def load_for_burstiness(timestamp_index, target_index, timestamp_pattern='%Y'):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'papers.xlsx'
    loc_stopwords = here / 'stopwordsEng.txt'

    # 데이터 로드
    df_papers = pd.read_excel(loc_data)
    df_papers.iloc[:, timestamp_index] = df_papers.iloc[:, timestamp_index].dt.strftime(timestamp_pattern)

    # 전처리
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Word(),
                            ptm.tagger.NLTK(),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),  # 품사 태크 제거
                            ptm.ngram.NGramTokenizer(1, 1),
                            ptm.helper.StopwordFilter(file=loc_stopwords))
    result = pipeline.processCorpus(df_papers.iloc[:, target_index])

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
    df_papers.iloc[:, target_index] = documents
    return df_papers

def load_for_bert(timestamp_index, target_index, timestamp_pattern='%Y', reuse_preproc=False):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'papers.xlsx'
    loc_stopwords = here / 'stopwordsEng.txt'

    # 기전처리된 파일 사용 시
    if reuse_preproc:
        with open(here / 'papers_pp_for_bert.pkl', 'rb') as fin:
            timestamps = pickle.load(fin)
            documents = pickle.load(fin)
        fin.close()
        return timestamps, documents
    # 데이터 로드
    df_papers = pd.read_excel(loc_data)
    timestamps = df_papers.iloc[:, timestamp_index].dt.strftime(timestamp_pattern).tolist()
    target = df_papers.iloc[:, [target_index]].astype(str).values.tolist()
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
    with open(here / 'papers_pp_for_bert.pkl', 'wb') as fout:
        pickle.dump(timestamps, fout)
        pickle.dump(documents, fout)
    fout.close()

    return timestamps, documents
