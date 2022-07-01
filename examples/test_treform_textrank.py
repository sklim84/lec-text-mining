import treform as ptm

_stopwords = []
with open("../stopwords/stopwordsKor.txt", encoding='utf-8') as file:
    for line in file:
        line = line.strip() #or some other preprocessing
        _stopwords.append(line) #storing everything in memory!

path='C:\\mecab\\mecab-ko-dic'
#pos_tagger_name - either komoran, okt, nltk
#lang = ko or en
pipeline = ptm.Pipeline(ptm.keyword.TextRankExtractor(pos_tagger_name='mecab',
                                                      mecab_path=path,
                                                      max=5,
                                                      lang='ko',
                                                      stopwords=_stopwords,
                                                      combined_keywords=True))

corpus = ptm.CorpusFromFile('../sample_data/sampleKor.txt')
result = pipeline.processCorpus(corpus)
print('== Splitting Sentence ==')
print(result)
print()

