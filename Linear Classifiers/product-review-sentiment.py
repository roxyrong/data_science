import json
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return str(text).translate(translator)


products = pd.read_csv('Linear Classifiers/amazon_baby.csv')
products = products.fillna({'review': ''})
products['review_clean'] = products['review'].apply(remove_punctuation)
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(lambda rating: 1 if rating > 3 else -1)

train_data, test_data = train_test_split(products, test_size=0.2)
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
test_matrix = vectorizer.transform(test_data['review_clean'])
train_target = train_data['sentiment']
test_target = test_data['sentiment']

regr1 = LogisticRegression()
regr1.fit(train_matrix, train_target)
test_predicted = regr1.predict(test_matrix)
test_target = np.array(test_target)
count = 0
for items in zip(test_predicted, test_target):
    if items[0] * items[1] == 1:
        count = count + 1
accuracy = count / len(test_target)


json_file = 'Linear Classifiers/important_words.json'
with open(json_file, 'r') as f:
    important_words = json.load(f)

vectorizer_word_subset = CountVectorizer(vocabulary=important_words)
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])

regr2 = LogisticRegression()
regr2.fit(train_matrix_word_subset, train_target)
test_predicted_subset = regr2.predict(test_matrix_word_subset)

count = 0
for items in zip(test_predicted_subset, test_target):
    if items[0] * items[1] == 1:
        count = count + 1
accuracy = count / len(test_target)