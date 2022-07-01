# pip install google-cloud-bigquery
# pip install -e git+https://github.com/SohierDane/BigQuery_Helper#egg=bq_helper
# key 파일 생성 : https://cloud.google.com/docs/authentication/getting-started#windows
# key 파일 Environment variable 추가 : GOOGLE_APPLICATION_CREDENTIALS="KEY_PATH"
from bq_helper import BigQueryHelper
import pandas as pd

patents = BigQueryHelper(active_project="patents-public-data", dataset_name="patents")
print(patents.list_tables())
query = '''SELECT 
  publication_number, abstract_localized
FROM
  `patents-public-data.patents.publications`
LIMIT
  10;'''
results = patents.query_to_pandas_safe(query)
print(results)
