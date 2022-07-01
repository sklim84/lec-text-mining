import treform as ptm
import re
import os
from study._datasets import jpss_data

####################
# Journal of Payments Strategy & Systems 논문
# - Format : id(eid) | year | title | abstract | keywords
# - 기간 : 2017-2021
# - 분석대상 : abstract
####################

# 데이터 로드 및 전처리
result = jpss_data.load(target=3)

# 구조 변경 : Sentence co-occurrence word를 찾기 위해 하나의 setence를 하나의 document로 변경
documents = []
for doc in result:
    for sent in doc:
        new_sent = ' '.join(sent)
        new_sent = re.sub('[^A-Za-z0-9가-힣_ ]+', '', new_sent)
        new_sent = new_sent.strip()
        if len(new_sent) > 0:
            documents.append(new_sent)

####################
# 두 방법으로 co-occurrence 계산 및 graphml 생성
# 1. CooccurrenceWorker : approximate co-occurrence(spase matrix -> dense matrix)
# 2. External Manager : count
####################

# 1-1. CooccurrenceWorker
co = ptm.cooccurrence.CooccurrenceWorker()
co_result, vocab = co(documents)

# 1-2. create graphml
graph_builder = ptm.graphml.GraphMLCreator()
graph_builder.createGraphML(co_result, vocab, "./results/jpss_w_cv.graphml")

# 2-1. External Manager
with open('./results/jpss_preprocess.txt', 'w', encoding='utf-8') as fout:
    for sent in documents:
        fout.write(sent + "\n")
fout.close()

# CooccurrenceExternalManager 내부에서 os.chdir()을 통해 path 변경
# 복원을 위해 현재 path 저장(복원하지 않을경우 program_path 값으로 설정되어 path 접근 불편)
current_path = os.getcwd()
co_occur = ptm.cooccurrence.CooccurrenceExternalManager(
    program_path=current_path + '\\external_programs',
    input_file='../results/jpss_preprocess.txt',
    output_file='../results/jpss_co_count.txt',
    threshold=1, num_workers=3)
co_occur.execute()
# path 복원
os.chdir(current_path)

# 2-2. create graphml
co_results = []
vocabulary = {}
with open('./results/jpss_co_count.txt', 'r', encoding='utf-8') as fin:
    for line in fin:
        fields = line.split()
        word1, word2, count = fields[0], fields[1], fields[2]
        tup = (' '.join([str(word1), str(word2)]), float(count))
        co_results.append(tup)
        vocabulary[word1] = vocabulary.get(word1, 0) + 1
        vocabulary[word2] = vocabulary.get(word2, 0) + 1
        word_hist = dict(zip(vocabulary.keys(), vocabulary.values()))

graph_builder.createGraphMLWithThreshold(co_results, word_hist, vocabulary.keys(),
                                         "./results/jpss_w_ext_th_10.graphml", threshold=10)
