# Spam filter with naive Beyes classifier

HAM = 0
SPAM = 1
datadir = 'data/chapter7'

sources = [
    ('beck-s.tar.gz', HAM),
    ('farmer-d.tar.gz', HAM),
    ('kaminski-v.tar.gz', HAM),
    ('kitchen-l.tar.gz', HAM),
    ('lokay-m.tar.gz', HAM),
    ('williams-w3.tar.gz', HAM),
    ('BG.tar.gz', SPAM),
    ('GP.tar.gz', SPAM),
    ('SH.tar.gz', SPAM)
]

def extract_tar(datafile, extractdir):
    try:
        import tarfile
    except ImportError:
        raise ImportError("You do not have tarfile installed")
    tar = tarfile.open(datafile)

    tar.extractall(path=extractdir)
    tar.close()
    print("%s successfully extracted to %s" % (datafile, extractdir))

for source, _ in sources:
    datafile = '%s/%s' % (datadir, source)
    extract_tar(datafile, datadir)

import os

def read_single_file(filename):
    past_header, lines = False, []
    if os.path.isfile(filename):
        f = open(filename, encoding='latin-1')
        for line in f:
            if past_header:
                lines.append(line)
            elif line == '\n':
                past_header = True
        f.close()
    content = '\n'.join(lines)
    return filename, content

def read_files(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            yield read_single_file(filepath)

import pandas as pd
pd.DataFrame({
    'model': ['Normal Bayes', 'Multinomial Bayes', 'Bernoulli Bayes'],
    'class': [
        'cv2.ml.NormalBayesClassifier_create()',
        'sklearn.naive_bayes.MultinomilNB()',
        'sklearn.naiev_bayes.BernoulliNB()'
    ]
})

def build_data_frame(extractdir, classification):
    rows = []
    index = []
    for file_name, text in read_files(extractdir):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)
    data_frame = pd.DataFrame(rows, index=index)
    return data_frame

data = pd.DataFrame({'text': [], 'class':[]})
for source, classificaiton in sources:
    extractdir = '%s/%s' % (datadir, source[:-7])
    data = data.append(build_data_frame(extractdir, classificaiton))

from sklearn import feature_extraction
counts = feature_extraction.text.CountVectorizer()
X = counts.fit_transform(data['text'].values)
X.shape

y = data['class'].values

from sklearn import model_selection as ms

X_train, X_test, y_train, y_test = ms.train_test_split(
    X, y, test_size=0.2, random_state=42
)

import cv2

model_norm = cv2.ml.NormalBayesClassifier_creat()

import numpy as np
X_train_small = X_train[:1000, :3000].toarray().astype(np.float32)
y_train_small = y_train[:1000].astype(np.float32)

model_norm.train(X_train_small, cv2.ml.ROW_SAMPLE, y_train_small)

from sklearn import naive_bayes
model_naive = naive_bayes.MultinomialNB()
model_naive.fit(X_train, y_train)

model_naive.score(X_train, y_train)
model_naive.score(X_test, y_test)

counts = feature_extraction.text.CountVectorizer(
    ngram_range=(1, 2)
)

X = counts.fit_transform(data['text'].values)

model_naive = naive_bayes.MultinomialNB()
model_naive.fit(X_train, y_train)

model_naive.score(X_test, y_test)

tfidf = feature_extraction.text.TfidfTransformer()
X_new = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = ms.train_test_split(
    X_new, y, test_size=0.3, random_state=42
)

model_naive = naive_bayes.MultinomialNB()

model_naive.fit(X_train, y_train)
model_naive.score(X_test, y_test)

from sklearn import metrics
metrics.confusion_matrix(y_test, model_naive.predict(X_test))
