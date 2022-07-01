import os
from os import path
import tomotopy as tp

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# from team_project._datasets import news_data, papers_data, patents_data
# from team_project.dmr.commons import dmr_model
#
# # Topic 수 별 DMR 모델 생성
# timestamps, dataset = news_data.load_for_dmr(timestamp_index=15, target_index=36, reuse_preproc=True)
#
# topic_numbers = list(range(100, 101, 2))
# for topic_number in topic_numbers:
#      print('##### topic number: {}'.format(topic_number))
#      model = dmr_model(dataset, timestamps, topic_number)
#      model.save('./models/news_topic_num_{}.model'.format(topic_number), full=True)

part = 'news'
result_file = './results/' + part + '_perplexity_coherence.csv'
# 기존 파일이 있는경우 append
columns = ['topic number', 'perplexity', 'coherence', 'metric']
if path.exists(result_file):
    df_result = pd.read_csv(result_file)
else:
    df_result = pd.DataFrame(columns=columns)

# perplexity/coherence 계산
topic_numbers = list(range(2, 101, 2))
coherence_metric = 'c_v'  # u_mass(0에 가까울수록 일관성 높음), c_uci, c_npmi, c_v(0과1사이, 0.55정도 수준)
perplexities = []
coherences = []
for topic_number in topic_numbers:
    model_name = './models/' + part + '_topic_num_' + str(topic_number) + '.model'
    model = tp.DMRModel.load(model_name)
    coherence = tp.coherence.Coherence(model, coherence=coherence_metric)
    print('topic num: {}\tperplexity: {}\tcoherence: {}'.format(topic_number, model.perplexity, coherence.get_score()))
    perplexities.append(model.perplexity)
    coherences.append(coherence.get_score())

# csv 저장
df_result = df_result.append(
    pd.DataFrame((zip(topic_numbers, perplexities, coherences, [coherence_metric] * len(topic_numbers))),
                 columns=columns), ignore_index=True)
print(df_result)
df_result.to_csv(result_file, index=False, encoding='utf-8-sig')

# plot perplexity/coherence
fig = make_subplots(specs=[[{"secondary_y": True}]])
trace1 = go.Scatter(x=df_result['topic number'], y=df_result['perplexity'], name='Perplexity')
trace2 = go.Scatter(x=df_result['topic number'], y=df_result['coherence'], name='Coherence')
fig.add_trace(trace1, secondary_y=False)
fig.add_trace(trace2, secondary_y=True)
fig.update_layout(title_text=part + ' perplexity and coherence')
fig.update_yaxes(title_text='perplexity', secondary_y=False)
fig.update_yaxes(title_text='coherence', secondary_y=True)
fig.update_xaxes(title_text='number of topics')
fig.write_html(os.path.splitext(result_file)[0] + '.html')
