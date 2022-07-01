# Scopus
# https://jehyunlee.github.io/2021/08/13/Python-DS-82-scopusapi/
# https://pybliometrics.readthedocs.io/en/stable/
# pip install pybliometrics
# api key : 826e9a4048e22045883318cc9167951b
from pybliometrics.scopus import ScopusSearch
import pandas as pd


query = 'TITLE-ABS-KEY(blockchain OR block-chain)' # 31523

# 최초 수행 시 download=True
scopus = ScopusSearch(query=query, download=False)
print(scopus.get_results_size())

paper_info_list = []
for index, paper in enumerate(scopus.results):
    if index % 100 == 0:
        print(index, paper.coverDate, paper.title)
    paper_info_list.append((paper.eid, paper.coverDate, paper.title, paper.description, paper.authkeywords))

dataset = pd.DataFrame(paper_info_list, columns=['eid', 'cover_date', 'title', 'abstract', 'keywords'])
# pip install openpyxl
dataset.to_excel('../_datasets/papers.xlsx'.format(scopus.get_results_size()), index=False, encoding='utf-8-sig')
