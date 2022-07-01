import treform as ptm

corpus = ptm.CorpusFromCSVFile('data/news_articles_201701_201812.csv', 4)

# Split(NLTK)
pipeline = ptm.Pipeline(ptm.splitter.NLTK())
print('### Original')
print(corpus.docs[:1])

result = pipeline.processCorpus(corpus.docs[:1])
print('\n### Split(NLTK)')
print(result)

# Split(NLTK) + Tokenize(MeCab)
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.MeCab('C:\mecab\mecab-ko-dic'))
result = pipeline.processCorpus(corpus.docs[:1])
print('\n### Split(NLTK) + Tokenize(MeCab)')
print(result)

# Split(NLTK) + Tokenize(MeCab) + POSFilter
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.MeCab('C:\mecab\mecab-ko-dic'),
                        ptm.helper.POSFilter('VV*'))
result = pipeline.processCorpus(corpus.docs[:1])
print('\n### Split(NLTK) + Tokenize(MeCab) + POSFilter')
print(result)

# Split(NLTK) + Tokenize(MeCab) + POSFilter + SelectWordOnly
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.MeCab('C:\mecab\mecab-ko-dic'),
                        ptm.helper.POSFilter('VV*'),
                        ptm.helper.SelectWordOnly())
result = pipeline.processCorpus(corpus.docs[:1])
print('\n### Split(NLTK) + Tokenize(MeCab) + POSFilter + SelectWordOnly')
print(result)

# Split(NLTK) + Tokenize(MeCab) + POSFilter + SelectWordOnly + WordCounter
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.MeCab('C:\mecab\mecab-ko-dic'),
                        ptm.helper.POSFilter('VV*'),
                        ptm.helper.SelectWordOnly(),
                        ptm.counter.WordCounter())
result = pipeline.processCorpus(corpus.docs[:1])
print('\n### Split(NLTK) + Tokenize(MeCab) + POSFilter + SelectWordOnly + WordCounter')
print(result)

pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.Word(),
                        ptm.ngram.NGramTokenizer(min=2, ngramCount=3, concat='_'),
                        ptm.helper.StopwordFilter(file='../../stopwords/stopwordsKor.txt'),
                        ptm.counter.WordCounter())
result = pipeline.processCorpus(corpus.docs[:1])
print('\n### NGram')
print(result)