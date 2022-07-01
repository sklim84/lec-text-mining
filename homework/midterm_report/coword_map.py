import treform as ptm
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
import datasets
from datasets.food_data import Target
import pathlib

####################
# 먹거리_식품_안전(2017~2012).txt
# - Format : 제목 | 날짜 | 언론기관 | 본문
# - 분석대상 : 본문
####################

# 파라미터 설정
target = Target.CONTENT  # 분석대상
ngram_min, ngram_max = 1, 1  # ngram 최소/최대
method = 'ext'  # cv : CountVectorizer, ext : external manager
co_threshold = 100  # 동시 출현 threshold 값

# 파라미터 설정에 따른 결과 디텍토리 이름 작성
result_dir = pathlib.Path(__file__).resolve().parent / 'results' / ('coword_' + '_'.join(
    [target.name.lower(), 'ngram_' + str(ngram_min) + '_' + str(ngram_max), method, 'th_' + str(co_threshold)]))
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

####################
# [수업내용]
# - tri-gram은 co-word 분석에서 적합하지 않은듯
# - key phrase를 대상으로 삼아도 좋을듯
# - 공기어 추출을 위해 대상어 선정 필요
#  - 대상어 : 네트워크 분석 시 주된 대상 어휘
#  - 분석 목적에 따라 결정 : 고빈도 어휘, 문헌 주제 잘 반영하는 중요 어휘 등
####################

documents = []
if not os.path.isfile(result_dir / 'preprocess.txt'):
    # 데이터 로드 및 전처리
    dataset = datasets.food_data.load_for_coword(target, ngram_min, ngram_max)
    # Sentence 내 co-occurrence 탐색을 위한 처리(sentence→document)
    with open(result_dir / 'preprocess.txt', 'w', encoding='utf-8') as fout:
        for doc in dataset:
            for sent in doc:
                new_sent = ' '.join(sent)
                new_sent = re.sub('[^A-Za-z0-9가-힣_ ]+', '', new_sent)
                new_sent = new_sent.strip()
                if len(new_sent) > 0:
                    documents.append(new_sent)
                    fout.write(new_sent + "\n")
    fout.close()
else:
    with open(result_dir / 'preprocess.txt', 'r', encoding='utf-8') as fin:
        for line in fin:
            documents.append(line)
    fin.close()

####################
# [수업내용] 관계 강도
#  - 의미적 관계성 : 단순 동시출현 빈도수가 아닌
#  - 토폴로지 유사성 : 계층적, 노드 or 에지 기준
#   - IC(Lin, 1998) : 두 개념 사이의 Least Common Subsumer(IS-A 계층에서 단어 관계 유추)
#  - 통계적 유사성 : 문헌집단을 통계적으로 모델링, 모델을 통해 유사도 유추
#   - LSA(Latent Semantic Analysis) : SVD 기반
#  - Word Embeddings
#   - Word2Vec(Mikolove et al., 2013) : 단어→벡터 변환, 단순히 앞뒤로 같이 나오는지
#                                       단어의 분포적 가설(주변에 위치하는 단어로 의미 이해)
#                                       단어 의미는 벡터로 인코딩
# ####################
co_results = []
vocab = []
word_hist = {}
if method == 'cv':
    # 제목/본문 대상 분석 시 메모리 오류 발생
    # - numpy.core._exceptions.MemoryError: Unable to allocate 60.3 TiB for an array with shape (2879798, 2879798) and data type float64
    cv = ptm.cooccurrence.CooccurrenceWorker()
    co_results, vocab = cv(documents)
    cv = CountVectorizer()
    cv_fit = cv.fit_transform(documents)
    word_list = cv.get_feature_names()
    count_list = cv_fit.toarray().sum(axis=0)
    word_hist = dict(zip(word_list, count_list))
elif method == 'ext':
    # CooccurrenceExternalManager 내부에서 os.chdir()을 통해 path 변경
    # 복원을 위해 현재 path 저장(복원하지 않을경우 program_path 값으로 설정되어 path 접근 불편)
    current_path = os.getcwd()
    ext = ptm.cooccurrence.CooccurrenceExternalManager(
        program_path=current_path + '\\external_programs',
        input_file=str(result_dir / 'preprocess.txt'),
        output_file=str(result_dir / 'count.txt'),
        threshold=co_threshold, num_workers=3)
    ext.execute()
    # path 복원
    os.chdir(current_path)

    vocabulary = {}
    with open(result_dir / 'count.txt', 'r', encoding='utf-8') as fin:
        for line in fin:
            fields = line.split()
            word1, word2, count = fields[0], fields[1], fields[2]
            tup = (' '.join([str(word1), str(word2)]), float(count))
            co_results.append(tup)
            vocabulary[word1] = vocabulary.get(word1, 0) + 1
            vocabulary[word2] = vocabulary.get(word2, 0) + 1
            word_hist = dict(zip(vocabulary.keys(), vocabulary.values()))
    vocab = vocabulary.keys()

####################
# [수업내용] 관계 강도 측정
# - t-score : O(관측빈도)-E(예상빈도) / root(O)
#             높을수록 대상어와 공기어 관계 ↑
####################
graph_builder = ptm.graphml.GraphMLCreator()
graph_builder.createGraphMLWithThreshold(co_results, word_hist, vocab,
                                         result_dir / 'map.graphml', threshold=co_threshold)
