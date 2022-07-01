import treform as ptm
import datasets
import os
import re
import json
import pathlib
from datasets.food_data import Target
import pandas as pd

####################
# 먹거리_식품_안전(2017~2012).txt
# - Format : 제목 | 날짜 | 언론사 | 본문
# - 분석대상 : 제목, 본문
####################

# 파라미터 설정에 따른 키워드 리스트 파일명 작성
target = Target.CONTENT  # 분석대상
method = 'tf_idf'  # tf_idf, tf_ig
num_terms = 100  # 출력 term 수

# 파라미터 설정에 따른 결과 디텍토리 이름 작성
result_dir = pathlib.Path(__file__).resolve().parent / 'results' / (
        'term_weights_' + '_'.join([target.name.lower(), method, 'top_' + str(num_terms)]))
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

####################
# 데이터 로드 및 전처리
# - title(제목) : label
# - content(본문) : 분석대상
####################
label, content = datasets.food_data.load_for_term_weighting(target)
print('label length : ' + str(len(label)))
print('content length : ' + str(len(content)))
documents = []
for index, doc in enumerate(content):
    document = ' '
    for sent in doc:
        for word in sent:
            new_word = re.sub('[^A-Za-z0-9가-힣_ ]+', '', word)
            if len(new_word) > 0:
                document += ' ' + word
    document = document.strip()
    if len(document) > 0:
        documents.append(document)
    else:
        print('remove {}th content '.format(str(index)))
        label.pop(index)   # label 제거

print('total num of label : ' + str(len(documents)))
print('total num of content : ' + str(len(documents)))

####################
# Term weighting
# - tf_idf : TF-IDF 기반
# - tf-ig : Information Gain 기반
####################
if method == 'tf_idf':
    tf_idf = ptm.weighting.TfIdf(documents, label_list=label)
    weights = tf_idf()
elif method == 'tf_ig':
    tf_ig = ptm.weighting.TfIg(documents, label_list=label)
    weights = tf_ig()

# Term weighting 값 기준 정렬 및 csv 저장
title_term_value_list = []
for title, term_value in weights.items():
    for term, value in term_value.items():
        title_term_value_list.append((title, term, value))
        print('{}\t{}\t{}'.format(title, term, str(value)))
df_results = pd.DataFrame(title_term_value_list, columns=['title', 'term', 'tf-idf'])
df_results.sort_values(by=['tf-idf'], axis=0, ascending=False, inplace=True)
df_results[:num_terms].to_csv(result_dir / 'term_weights.csv', index=False, encoding='utf-8-sig')
