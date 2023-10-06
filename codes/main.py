import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import classification_report
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import datetime
import pickle

# Dictionary Vectorizer
custom_vectorizer = DictVectorizer()
# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
# Count Vectorizer
count_vectorizer = CountVectorizer()
# Feature Hasher
feature_hasher = FeatureHasher(input_type="dict")

# get accuracy
def compare_with_test_set(correct_set,predicted_data):
    total = 0
    correct = 0
    for predicted_sentence, correct_sentence in zip(predicted_data, correct_set):
        for predicted_word, correct_word in zip(predicted_sentence, correct_sentence):
            total = total + 1
            if predicted_word[1] == correct_word[1]:
                correct = correct + 1

    accuracy = (correct / total) * 100
    return accuracy

# read_data() function from LR/new.py
def read_data(train_datapath,include_y=True):
    """Read sentences from given corpus data"""
    # Master list to hold all the sentences
    sentences = [] 
    # Read the corpus data
    with open(train_datapath) as f:
        content = f.readlines()
    # Create a local list to hold each sentence
    sentence = []
    for line in content:
        if line !='\n':
            # Remove the trailing newline character and split the line at whitespace
            line = line.strip() 
            # Get the word and pos tag
            word = line.split()[0]
            if include_y:
                word = word.lower()
                pos = ""
                # Get the pos tag
                pos = line.split()[1] 
                # Append the word and pos tag as a tuple to the local sentence
                sentence.append((word, pos)) 
            else:
                sentence.append(word)
        else:
            # Append the local sentence to the master list
            sentences.append(sentence)  
            sentence = []
    print("-> %d sentences are read from '%s'." % (len(sentences), train_datapath))
    return sentences

# Define a function to extract custom features
def word_features(sentence, index):
    word = sentence[index]
    return {
        'word': word,
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'prev_word': '' if index == 0 else sentence[index - 1],
        'prev_word2': '' if index <= 1 else sentence[index - 2],
        'prev_word3': '' if index <= 2 else sentence[index - 3],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'next_word2': '' if index >= len(sentence) - 2 else sentence[index + 2],
        'next-word3' : '' if index >= len(sentence) - 3 else sentence[index + 3],
        'word_length': len(word),  # Custom feature: word length
        'prefix-1': word[0],
        'prefix-2': word[:2],
        'prefix-3': word[:3],
        'suffix-1': word[-1],
        'suffix-2': word[-2:],
        'suffix-3': word[-3:],
        'has_hyphen': '-' in word,
        'is_numeric': word.isdigit(),
        'capitals_inside': word[1:].lower() != word[1:],
        'is_capitalized': word[0].upper() == word[0],
        'is_all_caps': word.upper() == word,
    }

def form_data(tagged_sentences, include_y=True):
    """Create datasets for training/evaluation/testing"""
    # Create a list to hold tuples of (features, label)
    data = []
    # Loop through each sentence and see if pos tags are to be included
    if include_y:
        for sentence in tagged_sentences:
            for i, (word, tag) in enumerate(sentence):
                features = word_features([w for w, _ in sentence], i)
                data.append((features, tag))
        return data
    else:
        for i, word in enumerate(tagged_sentences):
            features = word_features([w for w in tagged_sentences], i)
            data.append(features)
        return data 

def write_data(tagged_sentences, filename):
    # Write the tagged sentences to a file
    with open(filename, 'w') as f:
        for sentence in tagged_sentences:
            for i, (word, tag) in enumerate(sentence):
                f.write(word + ' ' + tag + '\n')
            f.write('\n')

def tag(sentence, classifier):
    """Tag single sentence"""
    # form the test data
    test_sentence = [word.lower() for word in sentence]
    test_data = form_data(test_sentence, include_y=False)
    # Extract features using the dictionary vectorizer
    X_test_custom = custom_vectorizer.transform(test_data)
    # Extract TF-IDF features using TfidfVectorizer
    X_test_text = [" ".join(x) for x in test_data]
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    # Extract Hash features using HashingVectorizer
    X_test_text_hash = feature_hasher.transform(test_data)
    # Combine TF-IDF and custom features
    X_combined_test = hstack([X_test_tfidf, X_test_custom])
    # Combine TF-IDF, custom and count features
    X_combined_test = hstack([X_combined_test, X_test_text_hash])
    # Predict the tags
    preds = classifier.predict(X_combined_test) 
    # Combine the words with the predicted tags
    tagged_sent = list(zip(sentence, preds))
    return tagged_sent

def tag_sents(sentences, classifier):
    """Tag multiple sentences"""
    tagged_sents = list()
    for sent in sentences:
        tagged_sents.append(tag(sent, classifier))
    return tagged_sents

def evaluate(test_sentences, classifier):
    """Evaluate the model"""
    # Form the test data
    test_data = form_data(test_sentences)
    # Extract features and labels from the dataset
    X_test, y_test = zip(*test_data)
    # Extract custom features using DictVectorizer
    X_test_custom = custom_vectorizer.transform(X_test)
    # Extract TF-IDF features using TfidfVectorizer
    X_test_text = [" ".join(x) for x in X_test]
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    # Extract Hash features using HashingVectorizer
    X_test_text_hash = feature_hasher.transform(X_test)
    # Combine TF-IDF and custom features
    X_combined_test = hstack([X_test_tfidf, X_test_custom])
    # Combine TF-IDF, custom and count features
    X_combined_test = hstack([X_combined_test, X_test_text_hash])
    # Predict the tags
    y_pred = classifier.predict(X_combined_test)
    # Print the classification report
    print("Classification Report for the Evaluation Data:\n")
    print(classification_report(y_test, y_pred))

def hyper_tuning(X_train, y_train, scores, estimator, parameters, cv, model_filename='lr-model-ht1.pkl'):
    print("# Estimator:",estimator)
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        classifier = GridSearchCV(estimator, parameters, cv=cv, scoring='%s' % score)
        print("Initializing training")
        classifier.fit(X_train, y_train)
        print("training complete")
        print("Best parameters set found on development set:")
        print(classifier.best_params_)
        print("Grid scores on development set:")
        means = classifier.cv_results_['mean_test_score']
        stds = classifier.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        # save model
        print('Saving model...')
        with open(model_filename, 'wb') as file:
            pickle.dump(classifier, file)
        return classifier
    
def init_training_with_cross_validation(X_train, y_train, filename):
    t_ini = datetime.datetime.now()
    print('Training...')
    lr_model = LogisticRegression(solver='liblinear', multi_class='auto', random_state=2)
    skf = StratifiedKFold(n_splits=5)
    scores = ['accuracy'] # can add scores like 'f1_macro', 'precision', 'recall'
    params = [{'C': [0.1, 1, 2, 3]}] # params for C
    lr_model = hyper_tuning(X_train, y_train, scores, lr_model, params, skf, filename)
    t_fin = datetime.datetime.now()
    print('Training completed in {} seconds'.format((t_fin - t_ini).total_seconds()))
    return lr_model

def main():
  
    # define the path to the training data
    TRAIN_DATA = "../dataset/train.txt"
    VAL_DATA = "../dataset/val_labelled.txt"
    VAL_DATA_UNLABELLED = "../dataset/val.txt"
    TEST_DATAPATH = "../dataset/test.txt"
    TEST_LABELLED = "../dataset/test_labelled.txt"
    FILE_NAME = '../dataset/lr-model-ht1.pkl'

    # Logistic Regression Classifier
    classifier = LogisticRegression(C=3, solver='liblinear', multi_class='auto', random_state=2)
    # Support Vector Classifier
    # classifier = SVC(C=1, kernel='linear', random_state=2)

    # Read the training data
    tagged_sentences = read_data(TRAIN_DATA)

    # Form the training data
    train_data = form_data(tagged_sentences)

    # Extract features and labels from the dataset
    X_train, y_train = zip(*train_data)

    # Vectorize the custom features using DictVectorizer
    X_train_custom = custom_vectorizer.fit_transform(X_train)

    # Extract TF-IDF features using TfidfVectorizer
    X_train_text = [" ".join(x) for x in X_train]  # Convert the list of words to sentences
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)

    # Extract Hash features using HashingVectorizer
    X_train_text_hash = feature_hasher.transform(X_train)

    # Combine TF-IDF and custom features
    X_combined_train = hstack([X_train_tfidf, X_train_custom])

    # Combine TF-IDF, custom and Hash features
    X_combined_train = hstack([X_combined_train, X_train_text_hash])

    # Train a logistic regression model
    classifier.fit(X_combined_train, y_train)

    # Hyperparameter tuning

    # classifier = init_training_with_cross_validation(X_combined_train, y_train, FILE_NAME)

    # load model
    # print('Loading model...')
    with open(FILE_NAME, 'rb') as file:
        classifier = pickle.load(file)

    # Evaluate the model on the test set    
    eval_test = read_data(VAL_DATA)
    evaluate(eval_test, classifier)

    # Test the model on the unlabelled set
    # correct_test_sen = read_data(TEST_LABELLED)
    # test_sentences = read_data(TEST_DATAPATH, False)
    correct_test_sen = read_data(VAL_DATA)
    test_sentences = read_data(VAL_DATA_UNLABELLED, False)
    predicted_data = tag_sents(test_sentences, classifier)
    print("Accuracy on test set: ", compare_with_test_set(correct_test_sen,predicted_data))

    # Write the tagged sentences to a file
    write_data(predicted_data, "../dataset/model_labelled.txt")

if __name__ == '__main__':
    main()
