import pandas as pd
import matplotlib.pyplot as plt

def showFrLine(data):
    plt.figure(figsize=(10.0, 6.0))
    plt.title('\'frequency * rank\' Line', fontsize=20)
    plt.ylabel('f * r', fontsize=15)
    plt.xlabel('r', fontsize=15)
    plt.yticks([])
    plt.xticks([])
    plt.plot(list(range(len(data))), data['Rank X Frequency'].tolist())
    plt.show()

# Load data
df_only_noun = pd.read_csv('./result/df_ONLY_NOUN.csv')
df_only_verb = pd.read_csv('./result/df_ONLY_VERB.csv')
df_all = pd.read_csv('./result/df_ALL.csv')
df_ngram = pd.read_csv('./result/df_ALL_W_NGRAM_WO_STOPWORDS.csv')

# Total Frequency/Words Graph
fig = plt.figure(figsize=(10.0, 6.0))
ax = fig.add_subplot()
ax.plot(list(range(len(df_only_noun))), df_only_noun['Frequency'], label='1-noun')
ax.plot(list(range(len(df_only_verb))), df_only_verb['Frequency'], label='2-verb')
ax.plot(list(range(len(df_all))), df_all['Frequency'], label='3-all')
ax.plot(list(range(len(df_ngram))), df_ngram['Frequency'], label='4-ngram')
ax.legend()
plt.title('Zipf\'s Law Graph')
plt.ylabel('Frequency')
plt.xlabel('Words')
plt.show()

# Minimum Frequency/Words Graph
x = min(len(df_only_noun), len(df_only_verb), len(df_all), len(df_ngram))
y = min(max(df_only_noun['Frequency']), max(df_only_verb['Frequency']),
        max(df_all['Frequency']), max(df_ngram['Frequency']))

fig = plt.figure(figsize=(10.0, 6.0))
ax = fig.add_subplot()
ax.plot(list(range(len(df_only_noun))), df_only_noun['Frequency'], label='1-noun')
ax.plot(list(range(len(df_only_verb))), df_only_verb['Frequency'], label='2-verb')
ax.plot(list(range(len(df_all))), df_all['Frequency'], label='3-all')
ax.plot(list(range(len(df_ngram))), df_ngram['Frequency'], label='4-ngram')
ax.legend()
ax.set_xlim([0, x])
ax.set_ylim([0, y])

plt.title('Zipf\'s Law Graph')
plt.ylabel('Frequency')
plt.xlabel('Words')
plt.show()

# Frequency * Rank Graph
sub_noun = df_only_noun[df_only_noun['Frequency'] > 1]
sub_verb = df_only_verb[df_only_verb['Frequency'] > 1]
sub_all = df_all[df_all['Frequency'] > 1]
sub_ngam = df_ngram[df_ngram['Frequency'] > 1]
showFrLine(sub_noun)
showFrLine(sub_verb)
showFrLine(sub_all)
showFrLine(sub_ngam)
