import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
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
    prefix = lambda s, n: s[:n] if len(s) > n else s
    suffix = lambda s, n: s[-n:] if len(s) > n else s

    return {
        'nb_terms': len(sentence),
        'word': word,
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': word[0].upper() == word[0] if word else False,
        'is_all_caps': word.upper() == word if word else False,
        'is_all_lower': word.lower() == word if word else False,
          'prefix-1': prefix(word, 1),
          'prefix-2': prefix(word, 2),
          'prefix-3': prefix(word, 3),
          'prefix-4': prefix(word, 4),
          'prefix-5': prefix(word, 5),
          'suffix-1': suffix(word, 1),
          'suffix-2': suffix(word, 2),
          'suffix-3': suffix(word, 3),
          'suffix-4': suffix(word, 4),
          'suffix-5': suffix(word, 5),
          'i-1_prefix-3': prefix(sentence[index-1], 3) if index > 0 else '',
          'i-1_suffix-3': suffix(sentence[index-1], 3) if index > 0 else '',
          'i+1_prefix-3': prefix(sentence[index+1], 3) if index < len(sentence) - 1 else '',
          'i+1_suffix-3': suffix(sentence[index+1], 3) if index < len(sentence) - 1 else '',
        'prev_word': sentence[index - 1] if index > 0 else '',
        'next_word': sentence[index + 1] if index < len(sentence) - 1 else '',
        'has_hyphen': '-' in word,
        'is_numeric': word.isdigit(),
        'capitals_inside': word[1:].lower() != word[1:]
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

def predict(sentence, classifier):
    """predict single sentence"""
    # form the test data
    test_sentence = [word.lower() for word in sentence]
    test_data = form_data(test_sentence, include_y=False)
    # Extract features using the dictionary vectorizer
    #X_test_custom = custom_vectorizer.transform(test_data)
    # Extract TF-IDF features using TfidfVectorizer
    X_test_text = [" ".join(x) for x in test_data]
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    X_test_count = count_vectorizer.transform(X_test_text)
    # Extract Hash features using HashingVectorizer
    X_test_text_hash = feature_hasher.transform(test_data)
    # Combine TF-IDF and custom features
    X_combined_test = hstack([X_test_count, X_test_tfidf])
    # Combine TF-IDF, custom and count features
    #X_combined_test = hstack([X_combined_test, X_test_text_hash])
    # Predict the tags
    preds = classifier.predict(X_combined_test) 
    # Combine the words with the predicted tags
    tagged_sent = list(zip(sentence, preds))
    return tagged_sent

def predict_sentences(sentences, classifier):
    """predict multiple sentences"""
    pos_tagged_sentences = list()
    for sent in sentences:
        pos_tagged_sentences.append(predict(sent, classifier))
    return pos_tagged_sentences

def evaluate(test_sentences, classifier):
    """Evaluate the model"""
    # Form the test data
    test_data = form_data(test_sentences)
    # Extract features and labels from the dataset
    X_test, y_test = zip(*test_data)
    # Extract custom features using DictVectorizer
    #X_test_custom = custom_vectorizer.transform(X_test)
    # Extract TF-IDF features using TfidfVectorizer
    X_test_text = [" ".join(x) for x in X_test]
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    X_test_count = count_vectorizer.transform(X_test_text)
    # Extract Hash features using HashingVectorizer
    X_test_text_hash = feature_hasher.transform(X_test)
    # Combine TF-IDF and custom features
    X_combined_test = hstack([X_test_count, X_test_tfidf])
    # Combine TF-IDF, custom and count features
    #X_combined_test = hstack([X_combined_test, X_test_text_hash])
    # Predict the tags
    y_pred = classifier.predict(X_combined_test)
    # Print the classification report
    print("Classification Report for the Evaluation Data:\n")
    print(classification_report(y_test, y_pred))

# hyperparameter tuning
def init_training_with_cross_validation(X_train, y_train, filename):
    t_ini = datetime.datetime.now()
    print('Training...')
    lr_model = MultinomialNB()
    skf = StratifiedKFold(n_splits=5)
    scores = ['accuracy']
    params = [{
    'C': [0.01, 0.1, 1.0, 10.0], # C is inverse of lambda
    'solver': ['liblinear', 'lbfgs'] # liblinear is L1, lbfgs is L2
    }]
    print("# Estimator:",lr_model)
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        classifier = GridSearchCV(lr_model, param_grid=params, cv=skf, scoring='%s' % score)
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
        with open(filename, 'wb') as file:
            pickle.dump(classifier, file)
    t_fin = datetime.datetime.now()
    print('Training completed in {} seconds'.format((t_fin - t_ini).total_seconds()))
    return classifier

# main function
def main():
  
    # define the path to the training data
    TRAIN_DATA = "../dataset/train.txt"
    TEST_DATAPATH = "../dataset/test.txt"
    FILE_NAME = '../dataset/lr-pos.pkl'

    # Initialize a logistic regression classifier
    classifier = MultinomialNB()

    # Read the training data
    tagged_sentences = read_data(TRAIN_DATA)

    # split the data into train and test
    train_data = tagged_sentences[:int(len(tagged_sentences)*0.8)]
    test_data = tagged_sentences[int(len(tagged_sentences)*0.8):]

    # Form the training data
    train_data = form_data(tagged_sentences)

    # Extract features and labels from the dataset
    X_train, y_train = zip(*train_data)

    # Vectorize the custom features using DictVectorizer
    #X_train_custom = custom_vectorizer.fit_transform(X_train)

    # Extract TF-IDF features using TfidfVectorizer
    X_train_text = [" ".join(x) for x in X_train]  # Convert the list of words to sentences
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_train_count = count_vectorizer.fit_transform(X_train_text)
    # Extract Hash features using HashingVectorizer
    #X_train_text_hash = feature_hasher.transform(X_train)

    # Combine TF-IDF and custom features
    #X_combined_train = hstack([X_train_tfidf, X_train_custom])

    # Combine TF-IDF, custom and Hash features
    X_combined_train = hstack([X_train_count, X_train_tfidf])

    # Hyperparameter tuning
    # classifier = init_training_with_cross_validation(X_combined_train, y_train, FILE_NAME)

    # Train a logistic regression model
    classifier.fit(X_combined_train, y_train)

    # save model
    print('Saving model...')
    with open(FILE_NAME, 'wb') as file:
       pickle.dump(classifier, file) 

    # load model
    print('Loading model...')
    with open(FILE_NAME, 'rb') as file:
        classifier = pickle.load(file)

    # Evaluate the model on the test set    
    print("Evaluating the model on the test set:")
    evaluate(test_data, classifier)

    # Predict on the unlabelled set
    print("Predicting on the unlabelled set:")
    test_sentences = read_data(TEST_DATAPATH, False)
    predicted_data = predict_sentences(test_sentences, classifier)

    # Write the tagged sentences to a file
    write_data(predicted_data, "../dataset/prediction-pioneers-lr.txt")

if __name__ == '__main__':
    main()