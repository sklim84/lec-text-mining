import pathlib
import treform as ptm
from enum import Enum

####################
# 먹거리_식품_안전(2017~2012).txt
# - Format : 제목 | 날짜 | 언론사 | 본문
####################

# 분석대상
class Target(Enum):
    TITLE = 0
    DATE = 1
    PRESS = 2
    CONTENT = 3

####################
# 키워드 추출을 위한 데이터 로드 및 전처리
# - 전처리 : Tokenizing, POS tagging & filtering, Stopword Filtering
####################
def load_for_keywords(target):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / '먹거리_식품_안전(2017~2012).txt'
    loc_stopwords = here / 'stopwordsKor.txt'

    # 데이터 로드
    corpus = ptm.CorpusFromFieldDelimitedFile(loc_data, target.value)
    # 전처리
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab('C:\\mecab\\mecab-ko-dic'),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.helper.StopwordFilter(file=loc_stopwords))
    return pipeline.processCorpus(corpus.docs)

####################
# Term Weighting을 위한 데이터 로드 및 처리
# - 전처리 : Tokenizing, POS tagging & filtering, Stopword Filtering
####################
def load_for_term_weighting(target):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / '먹거리_식품_안전(2017~2012).txt'
    loc_stopwords = here / 'stopwordsKor.txt'

    # 데이터 로드
    title = ptm.CorpusFromFieldDelimitedFile(loc_data, 0)
    content = ptm.CorpusFromFieldDelimitedFile(loc_data, target.value)
    # 전처리
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab('C:\\mecab\\mecab-ko-dic'),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),  # 품사 태크 제거
                            ptm.helper.StopwordFilter(file=loc_stopwords))
    return title.docs, pipeline.processCorpus(content)

####################
# 공기어 분석을 위한 데이터 로드 및 전처리
# - 전처리 : Tokenizing, POS tagging & filtering, n-gram, Stopword Filtering
####################
def load_for_coword(target, ngram_min, ngram_max):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / '먹거리_식품_안전(2017~2012).txt'
    loc_stopwords = here / 'stopwordsKor.txt'

    # 데이터 로드
    corpus = ptm.CorpusFromFieldDelimitedFile(loc_data, target.value)
    # 전처리
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab('C:\\mecab\\mecab-ko-dic'),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),  # 품사 태크 제거
                            ptm.ngram.NGramTokenizer(ngram_min, ngram_max),
                            ptm.helper.StopwordFilter(file=loc_stopwords))
    return pipeline.processCorpus(corpus)

