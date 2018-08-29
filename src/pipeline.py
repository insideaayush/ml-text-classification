from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

"""
Load Training and validation dataset
"""

# loading training data
training_dataset = pandas.read_csv("../ml-task-datasets/data_train.csv")
train_x = training_dataset["text"]
train_y = training_dataset["label"]

# loading validation data
validation_dataset = pandas.read_csv("../ml-task-datasets/data_dev.csv")
valid_x = validation_dataset["text"]
valid_y = validation_dataset["label"]

# encoding labels using label encoder
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

"""
Feature Extraction
"""

""" Count Vectors """

# create a count vectorizer object
count_vect_train_set = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect_train_set.fit(train_x)

count_vect_valid_set = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect_valid_set.fit(valid_x)

# transform the training and validation set using count vectorizer object
xtrain_count = count_vect_train_set.transform(train_x)
xvalid_count = count_vect_valid_set.transform(valid_x)

""" TF-IDF Vectors """
""" word level """
# create a tf-idf vectorizer
tfidf_vect_train_set = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect_train_set.fit(train_x)

tfidf_vect_valid_set = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect_valid_set.fit(valid_x)

# transform the training and validation set
xtrain_tfidf_word = tfidf_vect_train_set.transform(train_x)
xvalid_tfidf_word = tfidf_vect_valid_set.transform(valid_x)

""" ngram level """
# create a tf-idf vectorizer
tfidf_vect_ngram_train_set = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_train_set.fit(train_x)

tfidf_vect_ngram_valid_set = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_valid_set.fit(valid_x)

# transform the training and validation set
xtrain_tfidf_ngram = tfidf_vect_ngram_train_set.transform(train_x)
xvalid_tfidf_ngram = tfidf_vect_ngram_valid_set.transform(valid_x)

""" character level """
# create a tf-idf vectorizer
tfidf_vect_char_train_set = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_char_train_set.fit(train_x)

tfidf_vect_char_valid_set = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_char_valid_set.fit(valid_x)

# transform the training and validation set
xtrain_tfidf_char = tfidf_vect_char_train_set.transform(train_x)
xvalid_tfidf_char = tfidf_vect_char_valid_set.transform(valid_x)

""" 
Learning Model 
"""
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)

"""
SVM
"""
# accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=False)
# print("SVM, ngram tf-idf vectors:: ", accuracy)

""" Neural Network """

def create_model_architecture(input_size):
    # create input layer
    input_layer = layers.Input((input_size, ), sparse=True)

    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)

    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs=input_layer, outputs=output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return classifier

classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)
print("NN, n-gram tf-idf vectors :: ", accuracy)