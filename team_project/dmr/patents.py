import re

import pandas as pd
from treform.topic_model.pyTextMinerTopicModel import pyTextMinerTopicModel

from team_project._datasets import patents_data
from team_project.dmr.commons import dmr_model, topic_scoring, get_topic_labeler
import tomotopy as tp
import matplotlib.pyplot as plt
import numpy as np

#
topic_number = 12
# 기존 생성한 모델 재사용여부
reuse_trained_model = True

if reuse_trained_model:
    model = tp.DMRModel.load('./models/patents_topic_num_12.model')
else:
    # 데이터 로드 및 전처리
    timestamps, dataset = patents_data.load_for_dmr(timestamp_index=1, target_index=3, timestamp_pattern='%Y', reuse_preproc=True)

    # DMR 모델 학습 및 저장
    model = dmr_model(dataset, timestamps, topic_number)
    model.save('./models/patents.model', True)


# document별 dominant topic 정보 저장
topic_model_util = pyTextMinerTopicModel()
df_topic_sents_keywords, matrix = topic_model_util.format_topics_sentences(topic_number=topic_number, mdl=model)
# formatting
df_dominant_topic_for_each_doc = df_topic_sents_keywords.reset_index()
df_dominant_topic_for_each_doc.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic_for_each_doc.to_csv('./results/patents_dominent_topic_for_each_doc.csv', index=False,
                                      encoding='utf-8-sig')


# topic label, keyword 저장
labeler = get_topic_labeler(model)
df_topic_label_keyword = pd.DataFrame(columns=['topic number', 'label', 'keywords'])
for index, topic_number in enumerate(range(model.k)):
    label = ' '.join(label for label, score in labeler.get_topic_labels(topic_number, top_n=5))
    keywords = ' '.join(keyword for keyword, prob in model.get_topic_words(topic_number))
    df_topic_label_keyword.loc[index] = [topic_number, label, keywords]
df_topic_label_keyword.to_csv('./results/patents_topic_label_keyword.csv', index=False, encoding='utf-8-sig')


# timestamp별 topic score 계산 및 저장
df_topic_score = topic_scoring(model)
print(df_topic_score)
df_topic_score.to_csv('./results/patents_topic_score.csv', encoding='utf-8-sig')


# timestamp별 topic score line graph
df_topic_score.T.plot(style='.-', grid=True)
plt.title('Patent Topic Score')
ylim = max(abs(min(df_topic_score.min())), abs(max(df_topic_score.max()))) + 0.5
plt.ylim(-ylim, ylim)
plt.legend(loc='lower right', fontsize=8)
plt.savefig('./results/patents_topic_score.png')
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

df_probs.plot.barh(stacked=True)
plt.title('Patent Topic Distributions')
plt.legend(loc='lower right', fontsize=8)
plt.savefig('./results/patents_topic_dist.png')
plt.show()
