from team_project.burstiness.commons import compute_term_burstiness
from team_project.burstiness.commons import topic_modeling
from team_project.burstiness.commons import plot_bersty_terms
from team_project._datasets import patents_data
import pandas as pd

# 데이터 로드
df = patents_data.load_for_burstiness(timestamp_index=1, target_index=3)

short_ma_length = 6  # 단기 이동평균선
long_ma_length = 12  # 장기 이동평균선
signal_line_ma = 3  # 시그널 곡선 : N일 동안의 MACD 지수 이동평균
significance_ma_length = 3  #

# term burstiness 계산
burstiness, burstiness_over_time = compute_term_burstiness(df=df,
                                                           timestamp_index=1,
                                                           target_index=3,
                                                           long_ma_length=long_ma_length,
                                                           short_ma_length=short_ma_length,
                                                           significance_ma_length=significance_ma_length,
                                                           signal_line_ma=signal_line_ma)

# 시간별 term burstiness 계산 결과 csv 저장
burstiness_over_time.to_csv(
    './results/patents_bursty_terms_over_time_{}_{}_{}_{}.csv'.format(short_ma_length, long_ma_length, signal_line_ma,
                                                                   significance_ma_length),
    encoding='utf-8-sig')
# term bustiness 계산 결과 csv 저장
# - 결과 형식 : terms | max value | location(month)
burstiness.to_csv('./results/patents_bursty_terms_{}_{}_{}_{}.csv'.format(short_ma_length, long_ma_length, signal_line_ma,
                                                                       significance_ma_length),
                  encoding='utf-8-sig')

# topic modeling
num_topics, clusters, cluster_label, bursts, burstvectors, unique_time_stamp \
    = topic_modeling(burstiness, df=df, timestamp_index=1, target_index=3)

# save topics
df_clusters = pd.DataFrame(columns=['topic number', 'label', 'keywords'])
for index, key in enumerate(sorted(clusters.keys())):
    words = ' '.join(clusters[key])
    label = cluster_label[key]
    df_clusters.loc[index] = [key, label, words]
df_clusters.to_csv(
    './results/patents_bursty_clusters_{}_{}_{}_{}.csv'.format(short_ma_length, long_ma_length, signal_line_ma,
                                                            significance_ma_length), index=False, encoding='utf-8-sig')

# plot clusters
output_filename = './results/patents_bursty_clusters_{}_{}_{}_{}.png'.format(short_ma_length, long_ma_length,
                                                                          signal_line_ma, significance_ma_length)
plot_bersty_terms(output_filename, num_topics, clusters, cluster_label, bursts, burstvectors, unique_time_stamp)