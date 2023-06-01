from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import numpy as np

# Downloaded from https://www.kaggle.com/datasets/ozlerhakan/spam-or-not-spam-dataset
df = pd.read_csv('spam_or_not_spam.csv')
df = df[df['email'].notna() & df['label'].notna()]
df = df.drop_duplicates()
print('Preview of cleaned dataset')
print(df)


text = df['email'].tolist()
label = df['label'].tolist()

print('Count of each label')
print(df['label'].value_counts())

kf = KFold(n_splits=5, shuffle=True, random_state=64)

def perform_ml(mode):
    # k-fold cross validation
    print(f'Running using {mode}')
    i = 1
    for train_indexes, test_indexes in kf.split(text):
        # split data into train and test sets
        text_train, text_test = np.take(text, train_indexes), np.take(text, test_indexes)
        label_train, label_test = np.take(label, train_indexes), np.take(label, test_indexes)

        vectorizer = CountVectorizer()
        text_train_vectorized = vectorizer.fit_transform(text_train)

        # Train model
        if mode == 'logistic':
            model = LogisticRegression(max_iter=250)
            model.fit(text_train_vectorized, label_train)
        elif mode == 'svm':
            model = SVC(kernel='linear')
            model.fit(text_train_vectorized, label_train)

        # Evaluate model on testing data
        model_pred = model.predict(vectorizer.transform(text_test))

        # Show report
        print('========================================================')
        print(f'Report for fold #{i}')
        print(classification_report(label_test, model_pred))



        cm = confusion_matrix(label_test, model_pred, labels=[0, 1])
        print('Confusion matrix: ')
        print('   (predicted) 0 1')
        print('(actual) 0 ', cm[0], '\n(actual) 1 ', cm[1])
        print()
        i += 1
    print()

perform_ml('logistic')
perform_ml('svm')
