# https://pypi.org/project/patent-client/
from patent_client import Patent
import pandas as pd

# application 출원서
# application number 출원번호
# publication number 발행번호, 공고번호

# USPTO(United States Patent & Trademark Office)
# https://patft.uspto.gov/netahtml/PTO/index.html
# query = blockchain OR block-chain (7446)
#         blockchain (7345)
#         block-chain (291)
query = 'blockchain OR block-chain'
issue_date = '2008-01-01->2022-04-30'
patents = Patent.objects.filter(query=query, issue_date=issue_date)  # 7446
print(len(patents))

patent_info_list = []
err_count=0
for index, patent in enumerate(patents):
    try:
        pub = patent.publication
        if index % 100 == 0:
            print('Title\n- {} {}'.format(pub.publication_date, pub.title))
            print('Abstract\n- {}\n'.format(pub.abstract))
        patent_info_list.append((pub.publication_number, pub.publication_date, pub.title, pub.abstract))
    except IndexError as e:
        err_count += 1
        print(index, e)

dataset = pd.DataFrame(patent_info_list, columns=['publication number', 'publication date', 'title', 'abstract'])
# pip install openpyxl
dataset.to_excel('../_datasets/patents.xlsx', index=False, encoding='utf-8-sig')
