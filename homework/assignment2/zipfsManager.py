import treform as ptm
import platform

from nltk.probability import FreqDist
import re, operator
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

from enum import Enum

class SubAssignment(Enum):
    ONLY_NOUN = 1
    ONLY_VERB = 2
    ALL = 3
    ALL_W_NGRAM_WO_STOPWORDS = 4

def wordOnlyFDist(fdist):
    # only leave letters (does the rest count as "language"? tricky)
    word_only_keys = [k for k in fdist.keys() if re.search(r'^[가-힣a-zA-Z_]+$',k)]
    return ({key: fdist[key] for key in word_only_keys})

def getFontProperties():
    if str(platform.system()).lower().startswith('win'):
        # Window의 경우 폰트 경로
        font_path = 'C:/Windows/Fonts/malgun.ttf'
    elif str(platform.system()).lower().startswith('mac'):
        # for Mac
        font_path = '/Library/Fonts/AppleGothic.ttf'
    return fm.FontProperties(fname=font_path)

def countTermFrequency(source):
    term_counts = {}
    for doc in source:
        for sent in doc:
            for _str in sent:
                term_counts[_str[0]] = term_counts.get(_str[0], 0) + int(_str[1])
    return term_counts

def createPipeline(work_num):
    pipeline = None

    """
    Tokenizer
        kokoma : 성능 떨어짐
        okt : 품사 tag 상이(ex. NNS -> NNOUN)
        Hannanum : sentence parsing도 가능하지만 성능 떨어짐
        Komoran : 성능 괜찮음, 시간 오래걸림
        Mecab : 일본에서 제작, CRF(ML 알고리즘) 기반, 성능 좋고 빠름, C++ 기반
    """
    """ 
    POS
        명사(NN*) : 일반명사(NNG), 고유명사(NNP), 의존명사(NNB)
        동사(VV*) : 동사(VV), 동사+관형형 전성 어미(VV+ETM)
    """

    if work_num == SubAssignment.ONLY_NOUN:
        pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                                ptm.tokenizer.MeCab('C:\mecab\mecab-ko-dic'),
                                ptm.helper.POSFilter('NN*'),
                                ptm.helper.SelectWordOnly(),
                                ptm.counter.WordCounter())
    elif work_num == SubAssignment.ONLY_VERB:
        pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                                ptm.tokenizer.MeCab('C:\mecab\mecab-ko-dic'),
                                ptm.helper.POSFilter('VV*'),
                                ptm.helper.SelectWordOnly(),
                                ptm.counter.WordCounter())
    elif work_num == SubAssignment.ALL:
        pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                                ptm.tokenizer.MeCab('C:\mecab\mecab-ko-dic'),
                                ptm.helper.SelectWordOnly(),
                                ptm.counter.WordCounter())
    elif work_num == SubAssignment.ALL_W_NGRAM_WO_STOPWORDS:
        pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                                ptm.tokenizer.Word(),
                                ptm.ngram.NGramTokenizer(min=2, ngramCount=3, concat='_'),
                                ptm.helper.StopwordFilter(file='../../stopwords/stopwordsKor.txt'),
                                ptm.counter.WordCounter())
    return pipeline

def showHistogramWithTopFreqTerm(data, top):
    plt.figure(figsize=(10.0, 6.0))
    plt.title('Top-{} High Frequency Words'.format(top), fontsize=30)
    plt.xticks(fontproperties=getFontProperties(), fontsize=20, rotation='vertical')
    plt.yticks(fontsize=20)
    plt.ylabel('Frequency')
    plt.xlabel('Words')
    for index, row in data.head(top).iterrows():
        plt.bar(index, row['Frequency'])
    plt.show()

def showZipfsLawGraph(data):
    plt.figure(figsize=(10.0, 6.0))
    plt.title('Zipf\'s Law Graph', fontsize=20)
    plt.ylabel('Frequency', fontsize=15)
    plt.xlabel('Words', fontsize=15)
    plt.yticks([])
    plt.xticks([])
    plt.plot(list(range(len(data))), data['Frequency'])
    plt.show()

# 서브과제 번호
sub_assignment = SubAssignment.ONLY_NOUN

# Pipeline 생성
pipeline = createPipeline(sub_assignment)

# Sample Data 읽기
corpus = ptm.CorpusFromCSVFile('./data/news_articles_201701_201812.csv', 4)

# Pipeline 실행
result = pipeline.processCorpus(corpus.docs[:5000])

# Term Frequency 계산
term_freq = countTermFrequency(result)

# Term Frequency Filtering : Word Only
term_fdist = FreqDist()
for key, value in term_freq.items():
    term_fdist[key] += value
term_freq_word_only = wordOnlyFDist(term_fdist)

# Term Frequency Sorting
term_freq_sorted = sorted(term_freq_word_only.items(), key=operator.itemgetter(1), reverse=True)
print(len(term_freq_sorted))
# Zipf's Law data 생성
column_header = ['Rank', 'Frequency', 'Rank X Frequency']
df_zipf = pd.DataFrame(columns=column_header)
rank = 1
for pair in term_freq_sorted:
    df_zipf.loc[pair[0]] = [rank, pair[1], rank*pair[1]]
    rank = rank+1
    if rank%10000 == 0:
        print(rank)
print(df_zipf.head(10))

# Zipf's Law data  파일 출력
df_zipf.to_csv('./result/df_{}.csv'.format(sub_assignment.name), encoding='utf-8-sig')

# Top-10 High Frequency Words Histogram 출력
showHistogramWithTopFreqTerm(df_zipf, 10)

# Zipf's Law Graph 출력
showZipfsLawGraph(df_zipf)


