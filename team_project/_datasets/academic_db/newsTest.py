# https://newsapi.org/
# https://medium.com/rakuten-rapidapi/top-10-best-news-apis-google-news-bloomberg-bing-news-and-more-bbf3e6e46af6
# https://rapidapi.com/blog/rapidapi-featured-news-apis/?utm_source=google&utm_medium=cpc&utm_campaign=Beta&utm_term=%2Bnews%20%2Bapi_b&gclid=CjwKCAjwgr6TBhAGEiwA3aVuIQu79qhq3i4H45P6bwaqdLVQqneqTyK8cqoCNePZhjNSv6RCz3y5BRoCsWoQAvD_BwE
# api key : 0729eadc988a460baa6b10290d557c49
# python -m pip install newsapi-python
# Developer : Search articles up to a month old, 100 requests per day
from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key='0729eadc988a460baa6b10290d557c49')
newsapi.get_sources()
# /v2/everything
all_articles = newsapi.get_everything(q='blockchain',
                                      sources='bbc-news,the-verge',
                                      domains='bbc.co.uk,techcrunch.com',
                                      from_param='2022-04-02',
                                      to='2022-04-15',
                                      language='en',
                                      sort_by='relevancy',
                                      page=2)


for article in all_articles.get('articles'):
    print(article['title'])

