# http://biopython.org/DIST/docs/tutorial/Tutorial.html
# https://github.com/biopython/biopython
# pip install biopython
from Bio import Entrez
Entrez.email = 'captiong84@gmail.com'
handle = Entrez.esearch(db='pubmed',
                        sort='relevance',
                        retmax='100000', # up to a maximum of 100,000 records
                        retmode='xml',
                        term='blockchain OR \"block-chain\"',
                        mindate='2008',
                        maxdate='2022',
                        usehistory='y')
results = Entrez.read(handle)
print(len(results['IdList']))
handle = Entrez.efetch(db='pubmed',
                       retmode='xml',
                       id=results['IdList'])
papers = Entrez.read(handle)

for paper in papers['PubmedArticle']:
    print(paper)
    title = paper['MedlineCitation']['Article']['ArticleTitle']
    # MedlineCitation - DateCompleted
    # MedlineCitation - DateRevised
    # MedlineCitation - Article - ArticleDate
    # MedlineCitation - Article - Journal - JournalIssue - PubDate
    date = None
    if len(paper['MedlineCitation']['Article']['ArticleDate']) != 0:
        date = paper['MedlineCitation']['Article']['ArticleDate'][0]['Year'] \
               + paper['MedlineCitation']['Article']['ArticleDate'][0]['Month'] \
               + paper['MedlineCitation']['Article']['ArticleDate'][0]['Day']
    keywords = None
    if len(paper['MedlineCitation']['KeywordList']) != 0:
        keywords = '|'.join(paper['MedlineCitation']['KeywordList'][0])
    abstract = ' '.join(paper['MedlineCitation']['Article']['Abstract']['AbstractText'])

    print('Title\n- {}'.format(title))
    print('Date\n- {}'.format(date))
    print('Keywords\n- {}'.format(keywords))
    print('Abstract\n- {}\n'.format(abstract))
