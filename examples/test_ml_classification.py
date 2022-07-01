from treform.document_classification.ml_textclassification import documentClassifier
import treform as ptm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

def evaluate(document_classifier, y_test, y_pred, indices_test, model):
    # 6. evaluation
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=document_classifier.category_id_df.label.values, yticklabels=document_classifier.category_id_df.label.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # The vast majority of the predictions end up on the diagonal (predicted label = actual label),
    # where we want them to be. However, there are a number of misclassifications,
    # and it might be interesting to see what those are caused by:
    for predicted in document_classifier.category_id_df.category_id:
        for actual in document_classifier.category_id_df.category_id:
            if predicted != actual and conf_mat[actual, predicted] >= 10:
                print("'{}' predicted as '{}' : {} examples.".format(document_classifier.id_to_category[actual],
                                                                     document_classifier.id_to_category[predicted],
                                                                     conf_mat[actual, predicted]))
                print(document_classifier.df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][
                          ['label', 'text']])
                print('')

    # As you can see, some of the misclassified complaints are complaints that touch on more than one subjects
    # (for example, complaints involving both credit card and credit report). This sort of errors will always happen.
    # Again, we use the chi-squared test to find the terms that are the most correlated with each of the categories:
    model.fit(document_classifier.features, document_classifier.labels)
    N = 2

    # print(model.__class__.__name__.lower())

    # 8. final evaluation per class
    print(metrics.classification_report(y_test, y_pred, target_names=document_classifier.df['label'].unique()))


if __name__ == '__main__':
    document_classifier = documentClassifier()
    mecab_path = 'C:\\mecab\\mecab-ko-dic'
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab(mecab_path),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(2, 2),
                            # ptm.tokenizer.LTokenizerKorean(),
                            ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt'))

    ml_algorithms = ['RandomForestClassifier', 'LinearSVC', 'MultinomialNB', 'LogisticRegression', 'KNN',
                     'SGDClassifier', 'DecisionTreeClassifier', 'AdaBoostClassifier']
    # model_name = 0  -- RandomForestClassifier
    # model_name = 1  -- LinearSVC
    # model_name = 2  -- MultinomialNB
    # model_name = 3  -- LogisticRegression
    # model_name = 4  -- KNN
    # model_name = 5  -- SGDClassifier
    # model_name = 6 -- DecisionTreeClassifier
    # model_name = 7 -- AdaBoostClassifier
    model_index = 0
    model_name = ml_algorithms[model_index]

    # document category and id map
    id_category_json = '../models/ml_id_category.json'

    # mode is either train or predict
    mode = 'train'
    if mode is 'train':
        input_file = '../sample_data/3_class_naver_news.csv'
        # 1. text processing and representation
        corpus = ptm.CorpusFromFieldDelimitedFileForClassification(input_file, delimiter=',', doc_index=4, class_index=1, title_index=3)
        docs = corpus.docs
        tups = corpus.pair_map
        class_list = []
        for id in tups:
            # print(tups[id])
            class_list.append(tups[id])

        result = pipeline.processCorpus(corpus)
        print('==  ==')

        documents = []
        for doc in result:
            document = ''
            for sent in doc:
                document += " ".join(sent)
            documents.append(document)

        document_classifier.preprocess(documents, class_list, id_category_json=id_category_json)

        X_train, X_test, y_train, y_test, y_pred, indices_test, model = document_classifier.train(model_index=model_index)

        print('training is finished')

        evaluate(document_classifier, y_test, y_pred, indices_test, model)
        document_classifier.save(model, model_name='../models/' + model_name + '.model')
        document_classifier.saveVectorizer(model_name='../models/' + model_name + '_vectorizer.model')

    elif mode is 'predict':
        model = document_classifier.load('../models/' + model_name + '.model')
        vectorizer_model = document_classifier.loadVectorizer(
            model_name='../models/' + model_name + '_vectorizer.model')
        document_classifier.predict(model, vectorizer_model)

        # 7. prediction
        input = "../sample_data/navernews.txt"
        corpus = ptm.CorpusFromFieldDelimitedFile(input, 3)

        result = pipeline.processCorpus(corpus)
        print('==  ==')

        documents = []
        for doc in result:
            document = ''
            for sent in doc:
                document += " ".join(sent)
            documents.append(document)

        document_classifier.predict_realtime(model, vectorizer_model, documents, id_category_json=id_category_json)
