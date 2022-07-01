import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from treform.document_classification.ml_textclassification import documentClassifier

from _datasets.news_data import load_for_train, load_for_predict


def evaluate(document_classifier, y_test, y_pred, indices_test, model):
    # 6. evaluation
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=document_classifier.category_id_df.label.values,
                yticklabels=document_classifier.category_id_df.label.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('./results/' + model.__class__.__name__ + '_heatmap.png')
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

    # 8. final evaluation per class
    report = metrics.classification_report(y_test, y_pred, target_names=document_classifier.df['label'].unique(),
                                           output_dict=True)
    df_report = pd.DataFrame.from_dict(report)
    df_report.to_csv('./results/' + model.__class__.__name__ + '_report.csv')

    print(metrics.classification_report(y_test, y_pred, target_names=document_classifier.df['label'].unique()))


# Classifier 생성
classifier = documentClassifier()

# Machine larning model
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
model_file_name = './models/' + ml_algorithms[model_index] + '.model'
model_vectorizer_file_name = './models/' + ml_algorithms[model_index] + '_vectorizer.model'

# document category and id map
id_category_json = './models/ml_id_category.json'

# mode is either train or predict
mode = 'predict'
if mode is 'train':
    # 1. text processing and representation
    datasets, class_list = load_for_train()
    classifier.preprocess(datasets, class_list, id_category_json=id_category_json)

    print('training start')
    X_train, X_test, y_train, y_test, y_pred, indices_test, model = classifier.train(model_index=model_index)
    print('training is finished')

    evaluate(classifier, y_test, y_pred, indices_test, model)
    classifier.save(model, model_name=model_file_name)
    classifier.saveVectorizer(model_name=model_vectorizer_file_name)

elif mode is 'predict':

    # load prediction data
    datasets = load_for_predict()

    with open(id_category_json, 'r', encoding='utf-8') as handle:
        id_to_category = json.loads(handle.read())

    # navernews.txt 각 데이터의 실제 category
    true_categories = ['economy', 'IT_science', '사회', 'economy', '세계']

    columns = ['ml algorithm', 'news', 'real category', 'predicted category']
    df_predict = pd.DataFrame(columns=columns)
    for ml_algorithm in ml_algorithms:
        # load trained model
        model_file_name = './models/' + ml_algorithm + '.model'
        model_vectorizer_file_name = './models/' + ml_algorithm + '_vectorizer.model'
        model = classifier.load(model_file_name)
        vectorizer_model = classifier.loadVectorizer(model_name=model_vectorizer_file_name)

        # 7. prediction
        text_features = vectorizer_model.transform(datasets)
        predictions = model.predict(text_features)
        correct = 0
        for index, predicted in enumerate(predictions):
            real_category = true_categories[index]
            predicted_category = id_to_category[str(predicted)]
            if real_category == predicted_category:
                correct += 1
            df_predict = df_predict.append(
                pd.DataFrame(data=[[ml_algorithm, index, real_category, predicted_category]], columns=columns))
        print(ml_algorithm, correct / len(datasets))
    print(df_predict)
    df_predict.to_csv('./results/all_models_prediction_results.csv', index=False, encoding='utf-8-sig')
