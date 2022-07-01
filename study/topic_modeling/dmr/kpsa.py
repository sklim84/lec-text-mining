import re

import pandas as pd
from treform.topic_model.pyTextMinerTopicModel import pyTextMinerTopicModel

from study._datasets import kpsa_data
from study.topic_modeling.dmr.commons import dmr_model, topic_scoring, get_topic_labeler
import tomotopy as tp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 생성 토픽 수
topic_number = 10
# 기존 생성한 모델 재사용여부
reuse_trained_model = True

model = None
if reuse_trained_model:
    # 모델 로드
    model = tp.DMRModel.load('./models/kpsa.model')
else:
    # 데이터 로드 및 전처리
    timestamps, dataset = kpsa_data.load_for_dmr(timestamp_index=1, target_index=3)

    # DMR 모델 학습 및 저장
    model = dmr_model(dataset, timestamps, topic_number)
    model.save('./models/kpsa.model', True)


# document별 dominant topic 정보 저장
topic_model = pyTextMinerTopicModel()
df_topic_sents_keywords, matrix = topic_model.format_topics_sentences(topic_number=10, mdl=model)
# formatting
df_dominant_topic_for_each_doc = df_topic_sents_keywords.reset_index()
df_dominant_topic_for_each_doc.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic_for_each_doc.to_csv('./results/kpsa_dominent_topic_for_each_doc.csv', index=False,
                                      encoding='utf-8-sig')


# 토픽별 label, keyword 저장
labeler = get_topic_labeler(model)
df_topic_label_keyword = pd.DataFrame(columns=['topic number', 'label', 'keywords'])
for index, topic_number in enumerate(range(model.k)):
    label = ' '.join(label for label, score in labeler.get_topic_labels(topic_number, top_n=5))
    keywords = ' '.join(keyword for keyword, prob in model.get_topic_words(topic_number))
    df_topic_label_keyword.loc[index] = [topic_number, label, keywords]
df_topic_label_keyword.to_csv('./results/kpsa_topic_label_keyword.csv', index=False, encoding='utf-8-sig')


# timestamp별 topic score 계산 및 저장
df_topic_score = topic_scoring(model)
print(df_topic_score)
df_topic_score.to_csv('./results/kpsa_topic_score.csv', encoding='utf-8-sig')


# timestamp별 topic score line graph
matplotlib.rcParams['font.family'] ='Malgun Gothic'
df_topic_score.T.plot(style='.-', grid=True)
plt.title('Topic Score')
ylim = max(abs(min(df_topic_score.min())), abs(max(df_topic_score.max()))) + 0.5
plt.ylim(-ylim, ylim)
plt.legend(loc='lower right', fontsize=8)
plt.savefig('./results/kpsa_topic_score.png')
plt.show()


# timestamp별 topic distribution graph(using softmax)
probs = np.exp(model.lambdas - model.lambdas.max(axis=0))
probs /= probs.sum(axis=0)

df_probs = pd.DataFrame(data=probs).T
topic_label = []
labeler = get_topic_labeler(model)
for topic_number in range(model.k):
    label = ' '.join([label_tuple[0] for label_tuple in labeler.get_topic_labels(topic_number, top_n=2)])
    topic_label.append(label)
df_probs.columns = topic_label
df_probs.index = model.metadata_dict

matplotlib.rcParams['font.family'] ='Malgun Gothic'
df_probs.plot.barh(stacked=True)
plt.title('Topic Distributions')
plt.legend(loc='lower right', fontsize=8)
plt.savefig('./results/kpsa_topic_dist.png')
plt.show()
