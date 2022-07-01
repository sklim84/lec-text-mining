# https://github.com/MaartenGr/BERTopic

from bertopic import BERTopic
from study._datasets import kpsa_data

# 데이터 로드
timestamps, dataset = kpsa_data.load_for_bert(1, 3)

# 기존 생성한 모델 재사용여부
reuse_trained_model = False
if reuse_trained_model:
    topic_model = BERTopic.load('./models/kpsa.model')
else:
    topic_model = BERTopic(language='multilingual')  # select a model that supports 50+ languages.
    topics, probs = topic_model.fit_transform(dataset)
    topic_model.save('./models/kpsa.model')

# topic name, count, keyword 저장
df_topic_info = topic_model.get_topic_info()

dict_topics = topic_model.get_topics()
for topic_id in dict_topics:
    topic = dict_topics[topic_id]
    # [(keyword1, probs1), ......] (list(zip))→ [keyword1, ...] (join)→ keyword1 keyword2 ...
    keywords = ' '.join(list(zip(*topic))[0])
    df_topic_info.loc[df_topic_info['Topic'] == topic_id, 'Keywords'] = keywords
print(df_topic_info)
df_topic_info.to_csv('./results/kpsatopic_info.csv', encoding='utf-8-sig')

# topic keyword score bar chart
fig = topic_model.visualize_barchart()
fig.write_html("./results/kpsa_topic_keywords_score.html")

# topic similarity heatmap
fig = topic_model.visualize_heatmap()
fig.write_html("./results/kpsa_topic_similarity_heatmap.html")

# dynamic topic modeling (over time)
topics_over_time = topic_model.topics_over_time(dataset, topics, timestamps, nr_bins=20)
topics_over_time.to_csv('./results/kpsa_topic_over_time.csv')
fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=3)
fig.write_html("./results/kpsa_topic_over_time.html")
