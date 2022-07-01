import treform as ptm
import datasets
import os
import re
from datasets.food_data import Target
import pathlib

####################
# 먹거리_식품_안전(2017~2012).txt
# - Format : 제목 | 날짜 | 언론사 | 본문
# - 분석대상 : 제목, 본문
####################

# 파라미터 설정에 따른 키워드 리스트 파일명 작성
target = Target.CONTENT  # 분석대상
method = 'wr'  # wr : KRWordRank
num_words = 100  # 출력 keyword 수

# 파라미터 설정에 따른 결과 디텍토리 이름 작성
result_dir = pathlib.Path(__file__).resolve().parent / 'results' / (
            'keywords_' + '_'.join([target.name.lower(), method, 'top_' + str(num_words)]))
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

# 데이터 로드 및 전처리
dataset = datasets.food_data.load_for_keywords(target)

documents = []
for doc in dataset:
    document = ''
    for sent in doc:
        new_sent = ' '.join(sent)
        new_sent = re.sub('[^A-Za-z0-9가-힣_ ]+', '', new_sent)
        new_sent = new_sent.strip()
        document += new_sent
    documents.append(document)

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
    num_words=num_words)
keywords = keyword_extractor(documents)

with open(result_dir / 'keywords.txt', 'w', encoding='utf-8') as fout:
    for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
        print('{}\t{}\n'.format(word, r))
        fout.write('{}\t{}\n'.format(word, r))
fout.close()
