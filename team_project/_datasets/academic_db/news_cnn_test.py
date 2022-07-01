import newspaper
import transformers
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

link = 'https://www.cnn.com'
# Scans the webpage and finds all the links on it.
page_features = newspaper.build(link, language='en', memoize_articles=False)
# Initialize a list for article titles and text.
title_text = list()
# The page_features object contains article objects that are initialized with links to the web pages.
for article in page_features.articles:
    try:
        # Each article must be downloaded, then parsed individually.
        # This loads the text and title from the webpage to the object.
        article.download()
        article.parse()
        # Keep the text, title and URL from the article and append to a list.
        title_text.append({'title':article.title,
                           'body':article.text,
                           'url': article.url})
    except:
        # If, for any reason the download fails, continue the loop.
        print("Article Download Failed.")

# Save as a dataframe to avoid excessive calls on the web page.
articles_df = pd.DataFrame.from_dict(title_text)
articles_df.to_csv(r'CNN_Articles_Oct15_21.csv')