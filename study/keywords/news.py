import treform as ptm
from study._datasets import news_data
import re

####################
# 뉴스 분석(네이버 뉴스)
# - Format : date(yyyymm) | original date | press | title | link | content
# - 기간 : 2017-2021
# - 총 8119건
# - 분석대상 : content
####################

# 데이터 로드 및 전처리
dataset = news_data.load_for_keyword(target=5)

####################
# KRWordRank 기반 Keyword 추출
# - 파라미터 : min_count=5, max_length=10, beta=0.85, max_iter=10
#  - GitHub에 공유된 KRWordRank 테스트 코드의 설정값
#  - https://github.com/lovit/KR-WordRank/blob/master/tests/test_krwordrank.py
####################
keyword_extractor = ptm.keyword.KeywordExtractionKorean(
    min_count=5,  # 단어의 최소 출현 빈도수 (그래프 생성 시)
    max_length=10,  # 단어의 최대 길이
    beta=0.85,  # PageRank의 decaying factor beta
    max_iter=10,
    num_words=100)
keywords = keyword_extractor(dataset)

print(keywords)

with open('./results/news_keywords.txt', 'w', encoding='utf-8') as fout:
    for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
        print('{}\t{}\n'.format(word, r))
        fout.write('{}\t{}\n'.format(word, r))
fout.close()
