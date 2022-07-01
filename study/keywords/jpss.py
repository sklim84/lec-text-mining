import treform as ptm
from study._datasets import jpss_data
import re
import treform.keyword.textrank as tr

####################
# Journal of Payments Strategy & Systems 논문
# - Format : id(eid) | year | title | abstract | keywords
# - 기간 : 2017-2021
# - 분석대상 : abstract
####################

# 데이터 로드 및 전처리
dataset = jpss_data.load_for_keyword(target=3)


####################
# TextRank 기반 Keyword 추출
# - 파라미터 : pos_tagger_name='nltk', lang='en'
####################
keyword_extractor = tr.TextRank(pos_tagger_name='nltk', lang='en')
keyword_extractor.build_keywords(' '.join(dataset))
keywords = keyword_extractor.get_keywords(limit=100)

with open('./results/jpss_keywords.txt', 'w', encoding='utf-8') as fout:
    for word, r in keywords:
        print('{}\t{}\n'.format(word, r))
        fout.write('{}\t{}\n'.format(word, r))
fout.close()
