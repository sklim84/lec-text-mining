import os

import treform as ptm

from team_project._datasets import papers_data

# 데이터 로드 및 전처리
dataset = papers_data.load_for_coword(target_index=3)

# External Manager : count
with open('./results/papers_preprocess.txt', 'w', encoding='utf-8') as fout:
    for sent in dataset:
        fout.write(sent + "\n")
fout.close()

# CooccurrenceExternalManager 내부에서 os.chdir()을 통해 path 변경
# 복원을 위해 현재 path 저장(복원하지 않을경우 program_path 값으로 설정되어 path 접근 불편)
current_path = os.getcwd()
co_occur = ptm.cooccurrence.CooccurrenceExternalManager(
    program_path=current_path + '\\external_programs',
    input_file='../results/papers_preprocess.txt',
    output_file='../results/papers_co_count.txt',
    threshold=100, num_workers=3)
co_occur.execute()
# path 복원
os.chdir(current_path)

# create graphml
co_results = []
vocabulary = {}
with open('./results/papers_co_count.txt', 'r', encoding='utf-8') as fin:
    for line in fin:
        fields = line.split()
        word1, word2, count = fields[0], fields[1], fields[2]
        tup = (' '.join([str(word1), str(word2)]), float(count))
        co_results.append(tup)
        vocabulary[word1] = vocabulary.get(word1, 0) + 1
        vocabulary[word2] = vocabulary.get(word2, 0) + 1
        word_hist = dict(zip(vocabulary.keys(), vocabulary.values()))

graph_builder = ptm.graphml.GraphMLCreator()
graph_builder.createGraphMLWithThreshold(co_results, word_hist, vocabulary.keys(),
                                         "./results/papers_w_ext_th_100.graphml", threshold=100)
