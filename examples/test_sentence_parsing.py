
import nltk
import treform as ptm
from nltk.draw.tree import draw_trees
from nltk import tree, treetransforms
from copy import deepcopy

mecab_path='C:\\mecab\\mecab-ko-dic'

pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.MeCab(mecab_path),
                        ptm.syntactic_parser.BeneparSyntacticParser()
                        # ptm.syntactic_parser.NLTKRegexSyntacticParser()
                        )
corpus = ptm.CorpusByDataFrame('../sample_data/parser_sample.txt', '\t', 0, header=False)
#corpus = ptm.CorpusFromFieldDelimitedFile('../sample_data/parser_sample.txt', 0)
print(corpus.docs)

trees = pipeline.processCorpus(corpus)

for tree in trees:
    print(tree[0])
    t = nltk.Tree.fromstring(tree[0])
    draw_trees(t)
