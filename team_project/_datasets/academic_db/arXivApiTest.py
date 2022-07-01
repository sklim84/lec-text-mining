# https://github.com/lukasschwab/arxiv.py
# pip install arxiv
import arxiv

# ti:blockchain
# abs:blockchain
# ti:blockchain+OR+abs:blockchain
data = arxiv.Search(query='ti:blockchain OR abs:blockchain',
                    max_results=10,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending)

for paper in data.results():
    print('Title\n- {} {}'.format(paper.published.year, paper.title))
    print('Abstract\n- {}\n'.format(paper.summary))
    # paper.download_pdf(filename=str(paper.published.year) + ' ' + paper.title)